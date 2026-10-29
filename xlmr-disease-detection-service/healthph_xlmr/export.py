"""Prepare a validated inference bundle from an explicitly selected full run."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from .inference import DEFAULT_ARTIFACT_DIR, DiseasePredictor, inference_files, validate_artifacts


def export_model_bundle(source_dir: Path | str, target_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> Path:
    """Stage, load, and warm up before replacement. Stop the service before updating."""
    source = Path(source_dir).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and target bundle directories must not overlap")
    service_root = Path(__file__).resolve().parents[1]
    if (target == service_root or service_root.is_relative_to(target)
            or (target.is_relative_to(service_root) and target != service_root / "model")):
        raise ValueError("Target must be the service model directory or a separate external model directory")
    validate_artifacts(source)
    if target.exists() and not target.is_dir():
        raise ValueError("Target must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=".xlmr-export-", dir=target.parent))
    preserve_backup = False
    try:
        stage = temporary / "bundle"
        for name in inference_files(source):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        predictor = DiseasePredictor(stage)
        predictor.predict("disease inference export check")
        del predictor
        backup = temporary / "previous"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except BaseException:
            if backup.exists():
                try:
                    os.replace(backup, target)
                except BaseException as exc:
                    preserve_backup = True
                    raise RuntimeError(f"Replacement and rollback failed; previous bundle preserved at {backup}") from exc
            raise
    finally:
        if not preserve_backup:
            shutil.rmtree(temporary)
    return target


def main():
    parser = argparse.ArgumentParser(description="Export a completed full XLM-RoBERTa notebook run")
    parser.add_argument("source_dir", help="Saved run directory containing run_config.json and best_model/")
    parser.add_argument("--target-dir", default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()
    print(f"Exported XLM-RoBERTa bundle: {export_model_bundle(args.source_dir, args.target_dir)}")


if __name__ == "__main__":
    main()
