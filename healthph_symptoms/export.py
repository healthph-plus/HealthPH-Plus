"""Export standalone disease services with keyword symptom extraction."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp

from .keywords import MASTER_KEYWORDS

PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_FILES = ("__init__.py", "keywords.py", "aliases.json")


def _replace_directory(stage: Path, target: Path) -> None:
    """Replace while stopped; preserve the previous directory if rollback fails."""
    backup = stage.parent / "previous"
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(stage, target)
    except BaseException:
        if backup.exists():
            try:
                os.replace(backup, target)
            except BaseException as exc:
                raise RuntimeError(f"Replacement and rollback failed; previous bundle preserved at {backup}") from exc
        raise


def _stage_runtime(target: Path) -> None:
    target.mkdir(parents=True)
    for name in RUNTIME_FILES:
        shutil.copy2(PACKAGE_DIR / name, target / name)
    shutil.copy2(MASTER_KEYWORDS, target / "keywords_master.csv")
    (target / "GENERATED.txt").write_text(
        "Generated from the repository healthph_symptoms package. Do not edit this copy.\n"
        "Refresh with: python -m healthph_symptoms.export --service-dir <folder> --code-only\n",
        encoding="utf-8",
    )


@contextmanager
def _runtime_transaction(service: Path):
    """Restore both runtime and workflow if a subsequent model switch fails."""
    temporary = Path(mkdtemp(prefix=".symptom-code-", dir=service))
    preserve, touched = False, []
    try:
        _stage_runtime(temporary / "runtime")
        shutil.copy2(PACKAGE_DIR / "KEYWORD_WORKFLOW.md", temporary / "workflow")
        for staged, name in (("runtime", "healthph_symptoms"), ("workflow", "SYMPTOM_WORKFLOW.md")):
            destination, backup = service / name, temporary / ("previous_" + name)
            if destination.exists():
                os.replace(destination, backup)
            touched.append((name, backup))
            os.replace(temporary / staged, destination)
        yield
    except BaseException:
        try:
            for name, backup in reversed(touched):
                destination = service / name
                if destination.exists():
                    os.replace(destination, temporary / ("failed_" + name))
                if backup.exists():
                    os.replace(backup, destination)
        except BaseException as exc:
            preserve = True
            raise RuntimeError(f"Runtime rollback failed; previous bundle preserved at {temporary}") from exc
        raise
    finally:
        if not preserve:
            shutil.rmtree(temporary)


def vendor_runtime(service_dir: Path | str) -> Path:
    """Publish the canonical runtime, without creating or faking model weights."""
    service = Path(service_dir).resolve()
    _service_module(service)
    with _runtime_transaction(service):
        pass
    return service / "healthph_symptoms"


def _service_module(service: Path) -> str:
    if (service / "healthph_xlmr/inference.py").is_file():
        return "healthph_xlmr.inference"
    if (service / "healthph_plus/disease_inference.py").is_file():
        return "healthph_plus.disease_inference"
    raise ValueError(f"Not a HealthPH disease service folder: {service}")


def _copy_disease_bundle(source: Path, stage: Path, module: str) -> None:
    if not (source / "best_model").is_dir():
        raise ValueError("Disease source requires a best_model directory")
    names = ["label_order.json", "thresholds.json"]
    if module.startswith("healthph_xlmr"):
        names.append("run_config.json")
    for name in names:
        shutil.copy2(source / name, stage / name)
    # Explicit tokenizer/model assets; optimizer/trainer state and datasets never
    # enter the developer handoff.
    model_files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files += [name for name in (
        "sentencepiece.bpe.model", "special_tokens_map.json", "added_tokens.json", "vocab.txt",
    ) if (source / "best_model" / name).is_file()]
    (stage / "best_model").mkdir()
    for name in model_files:
        shutil.copy2(source / "best_model" / name, stage / "best_model" / name)


def warmup_service_bundle(service: Path, bundle: Path, runtime_parent: Path) -> dict:
    """Load staged assets in an isolated interpreter using staged shared code."""
    module = _service_module(service)
    script = (
        "import importlib,json,sys; "
        "sys.path[:0]=[sys.argv[1],sys.argv[2]]; "
        "m=importlib.import_module(sys.argv[3]); "
        "p=m.DiseasePredictor(sys.argv[4]); "
        "r=p.predict('ubo at sipon'); "
        "assert isinstance(r['symptoms'],list) and r['symptom_model_version']; "
        "print(json.dumps({'model_version':r['model_version'],"
        "'symptom_model_version':r['symptom_model_version']}))"
    )
    env = {key: value for key, value in os.environ.items()
           if key != "HEALTHPH_SYMPTOM_KEYWORDS_PATH"}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(runtime_parent), str(service), module, str(bundle)],
        env=env, capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode:
        raise ValueError(f"Staged combined bundle failed offline startup: {result.stderr[-5000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def export_service_bundle(service_dir: Path | str, disease_source: Path | str,
                          target_dir: Path | str | None = None) -> Path:
    """Validate complete assets before replacement; stop the service first."""
    service = Path(service_dir).resolve()
    module = _service_module(service)
    disease = Path(disease_source).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve() if target_dir is not None else service / "model"
    if (target == service or service.is_relative_to(target)
            or (target.is_relative_to(service) and target != service / "model")
            or target == PACKAGE_DIR or target.is_relative_to(PACKAGE_DIR)
            or PACKAGE_DIR.is_relative_to(target)):
        raise ValueError("Target must be the service model directory or a separate external model directory")
    if disease == target or disease.is_relative_to(target) or target.is_relative_to(disease):
        raise ValueError("Source and target bundle directories must not overlap")
    if target.exists() and not target.is_dir():
        raise ValueError("Target must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=".symptom-export-", dir=target.parent))
    preserve = False
    try:
        stage = temporary / "bundle"
        stage.mkdir()
        _copy_disease_bundle(disease, stage, module)
        runtime_parent = temporary / "runtime"
        _stage_runtime(runtime_parent / "healthph_symptoms")
        warmup_service_bundle(service, stage, runtime_parent)
        # Publish source before the asset switch while the service is stopped.
        # A model-validation failure above leaves the existing handoff untouched.
        with _runtime_transaction(service):
            _replace_directory(stage, target)
    except RuntimeError as exc:
        preserve = "previous bundle preserved" in str(exc)
        raise
    finally:
        if not preserve:
            shutil.rmtree(temporary)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--disease-source")
    parser.add_argument("--target-dir")
    parser.add_argument("--code-only", action="store_true",
                        help="Refresh generated runtime code only; no model is trained or installed")
    args = parser.parse_args(argv)
    if args.code_only:
        if args.disease_source or args.target_dir:
            parser.error("--code-only cannot be combined with model paths")
        print(vendor_runtime(args.service_dir))
    else:
        if not args.disease_source:
            parser.error("--disease-source is required for export")
        print(export_service_bundle(args.service_dir, args.disease_source, args.target_dir))


if __name__ == "__main__":
    main()
