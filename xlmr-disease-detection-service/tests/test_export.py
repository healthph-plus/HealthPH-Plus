import json
from pathlib import Path
import shutil

import pytest

from healthph_xlmr.export import export_model_bundle
from healthph_xlmr.inference import REQUIRED_FILES, artifact_fingerprint


@pytest.fixture
def staged_loader(monkeypatch):
    import healthph_xlmr.export as export
    calls = []

    class FakePredictor:
        def __init__(self, root):
            assert all((root / name).is_file() for name in REQUIRED_FILES)
            calls.append(root)

        def predict(self, text):
            calls.append(text)
            return {}

    monkeypatch.setattr(export, "DiseasePredictor", FakePredictor)
    return calls


def test_export_excludes_training_state_and_replaces_complete_bundle(bundle, tmp_path, staged_loader):
    (bundle / "best_model/training_args.bin").write_bytes(b"training")
    (bundle / "best_model/sentencepiece.bpe.model").write_text("optional tokenizer asset")
    (bundle / "reports").mkdir()
    (bundle / "reports/test_predictions.csv").write_text("private training data")
    target = tmp_path / "model"
    target.mkdir()
    (target / "stale-file").write_text("previous")
    assert export_model_bundle(bundle, target) == target
    assert len(staged_loader) == 2
    assert not staged_loader[0].exists()
    assert not (target / "stale-file").exists()
    assert sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()) == sorted([
        *REQUIRED_FILES, "best_model/sentencepiece.bpe.model",
    ])
    assert artifact_fingerprint(target) == artifact_fingerprint(bundle)


def test_failed_warmup_keeps_previous_bundle(bundle, tmp_path, monkeypatch):
    import healthph_xlmr.export as export

    target = tmp_path / "model"
    shutil.copytree(bundle, target)
    before = artifact_fingerprint(target)
    (bundle / "thresholds.json").write_text('{"AURI": 0.7, "PN": 0.5, "TB": 0.5, "COVID": 0.5}')

    class BrokenPredictor:
        def __init__(self, root, **kwargs):
            pass

        def predict(self, text):
            raise ValueError("bad weights")

    monkeypatch.setattr(export, "DiseasePredictor", BrokenPredictor)
    with pytest.raises(ValueError, match="bad weights"):
        export_model_bundle(bundle, target)
    assert artifact_fingerprint(target) == before
    assert not list(tmp_path.glob(".xlmr-export-*"))


def test_failed_replacement_restores_previous_bundle(bundle, tmp_path, staged_loader, monkeypatch):
    import healthph_xlmr.export as export

    target = tmp_path / "model"
    shutil.copytree(bundle, target)
    before = artifact_fingerprint(target)
    (bundle / "thresholds.json").write_text('{"AURI": 0.7, "PN": 0.5, "TB": 0.5, "COVID": 0.5}')
    original = export.os.replace

    def replace(source, destination):
        if source.name == "bundle":
            raise OSError("replacement failure")
        return original(source, destination)

    monkeypatch.setattr(export.os, "replace", replace)
    with pytest.raises(OSError, match="replacement failure"):
        export_model_bundle(bundle, target)
    assert artifact_fingerprint(target) == before


def test_reject_overlapping_paths(bundle):
    for target in [bundle, bundle / "model", bundle.parent]:
        with pytest.raises(ValueError, match="overlap"):
            export_model_bundle(bundle, target)


def test_failed_rollback_preserves_backup_for_manual_recovery(bundle, tmp_path, staged_loader, monkeypatch):
    import healthph_xlmr.export as export

    target = tmp_path / "model"
    shutil.copytree(bundle, target)
    before = artifact_fingerprint(target)
    original = export.os.replace

    def replace(source, destination):
        if source.name in {"bundle", "previous"}:
            raise OSError("replacement failure")
        return original(source, destination)

    monkeypatch.setattr(export.os, "replace", replace)
    with pytest.raises(RuntimeError, match="previous bundle preserved"):
        export_model_bundle(bundle, target)
    backup = next(tmp_path.glob(".xlmr-export-*/previous"))
    assert artifact_fingerprint(backup) == before


def test_smoke_export_fails_before_touching_target(bundle, tmp_path):
    config_path = bundle / "run_config.json"
    config = json.loads(config_path.read_text())
    config.update(run_mode="smoke", smoke_only=True)
    config_path.write_text(json.dumps(config))
    target = tmp_path / "model"
    with pytest.raises(ValueError, match="Smoke"):
        export_model_bundle(bundle, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".xlmr-export-*"))


def test_reject_file_target(bundle, tmp_path):
    target = tmp_path / "model"
    target.write_text("keep this file")
    with pytest.raises(ValueError, match="Target must be a directory"):
        export_model_bundle(bundle, target)
    assert target.read_text() == "keep this file"


def test_export_cannot_replace_service_source_tree(bundle):
    import healthph_xlmr.export as export

    service = Path(export.__file__).resolve().parents[1]
    for target in (service, service / "healthph_xlmr", service.parent):
        with pytest.raises(ValueError, match="Target must be"):
            export_model_bundle(bundle, target)
