import shutil

import pytest

from healthph_sentiment.export import export_model_bundle
from healthph_sentiment.inference import REQUIRED_FILES, artifact_fingerprint
from test_inference import bundle


@pytest.fixture
def staged_loader(monkeypatch):
    import healthph_sentiment.export as export
    calls = []

    class FakePredictor:
        def __init__(self, root):
            assert all((root / name).is_file() for name in REQUIRED_FILES)
            calls.append(root)

        def predict(self, text):
            calls.append(text)
            return {}

    monkeypatch.setattr(export, "SentimentPredictor", FakePredictor)
    return calls


def test_export_excludes_training_state_and_replaces_complete_bundle(bundle, tmp_path, staged_loader):
    (bundle / "best_model/training_args.bin").write_bytes(b"training")
    (bundle / "best_model/vocab.txt").write_text("optional tokenizer asset")
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
        *REQUIRED_FILES, "best_model/vocab.txt",
    ])
    assert artifact_fingerprint(target) == artifact_fingerprint(bundle)


def test_failed_warmup_keeps_previous_bundle(bundle, tmp_path, monkeypatch):
    import healthph_sentiment.export as export

    target = tmp_path / "model"
    shutil.copytree(bundle, target)
    before = artifact_fingerprint(target)
    (bundle / "threshold.json").write_text('{"worried_threshold": 0.7}')

    class BrokenPredictor:
        def __init__(self, root):
            pass

        def predict(self, text):
            raise ValueError("bad weights")

    monkeypatch.setattr(export, "SentimentPredictor", BrokenPredictor)
    with pytest.raises(ValueError, match="bad weights"):
        export_model_bundle(bundle, target)
    assert artifact_fingerprint(target) == before
    assert not list(tmp_path.glob(".sentiment-export-*"))


def test_failed_replacement_restores_previous_bundle(bundle, tmp_path, staged_loader, monkeypatch):
    import healthph_sentiment.export as export

    target = tmp_path / "model"
    shutil.copytree(bundle, target)
    before = artifact_fingerprint(target)
    (bundle / "threshold.json").write_text('{"worried_threshold": 0.7}')
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
