import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from healthph_plus.disease_inference import (
    LABELS, MAX_LENGTH, REQUIRED_FILES, DiseasePredictor,
    artifact_fingerprint, results_from_logits, validate_artifacts, validate_text,
)


@pytest.fixture
def bundle(tmp_path):
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "label_order.json").write_text(json.dumps(list(LABELS)))
    (tmp_path / "thresholds.json").write_text(json.dumps(dict.fromkeys(LABELS, 0.5)))
    (tmp_path / "best_model/config.json").write_text(json.dumps({
        "model_type": "electra", "problem_type": "multi_label_classification",
        "id2label": {str(i): label for i, label in enumerate(LABELS)},
        "label2id": {label: i for i, label in enumerate(LABELS)},
    }))
    return tmp_path


@pytest.mark.parametrize("text", [None, 1, True, [], {}, "", " \n\t", "a" * 10_001])
def test_invalid_text(text):
    with pytest.raises(ValueError):
        validate_text(text)


def test_text_is_not_normalized():
    text = "  Ubo\n at lagnat — ñ 😷 #health https://example.com  "
    assert validate_text(text) == text
    assert validate_text("a" * 10_000) == "a" * 10_000


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_missing_artifacts_fail(bundle, name):
    (bundle / name).unlink()
    with pytest.raises(ValueError, match="Missing disease inference artifacts"):
        DiseasePredictor(bundle)


@pytest.mark.parametrize("value", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_thresholds(bundle, value):
    thresholds = dict.fromkeys(LABELS, 0.5)
    thresholds["TB"] = value
    (bundle / "thresholds.json").write_text(json.dumps(thresholds))
    with pytest.raises(ValueError, match="Invalid probability threshold"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("thresholds", [{}, {"AURI": 0.5}, dict.fromkeys((*LABELS, "OTHER"), 0.5), []])
def test_threshold_labels_must_match(bundle, thresholds):
    (bundle / "thresholds.json").write_text(json.dumps(thresholds))
    with pytest.raises(ValueError, match="exactly AURI"):
        validate_artifacts(bundle)


def test_label_order_must_match(bundle):
    (bundle / "label_order.json").write_text(json.dumps(list(reversed(LABELS))))
    with pytest.raises(ValueError, match="in that order"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("key,value", [
    ("model_type", "bert"), ("problem_type", "single_label_classification"),
    ("id2label", {"0": "COVID"}), ("label2id", {"COVID": 0}), ("num_labels", 3),
])
def test_inconsistent_model_config(bundle, key, value):
    path = bundle / "best_model/config.json"
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        validate_artifacts(bundle)


def test_fingerprint_is_portable_and_ignores_training_files(bundle, tmp_path_factory):
    original = artifact_fingerprint(bundle)
    target = tmp_path_factory.mktemp("relocated") / "bundle"
    shutil.copytree(bundle, target)
    assert artifact_fingerprint(target) == original
    (bundle / "best_model_info.json").write_text('{"model_dir":"/another/machine"}')
    (bundle / "trainer_output").mkdir()
    (bundle / "trainer_output/state.json").write_text("{}")
    (bundle / "best_model/.DS_Store").write_bytes(b"finder metadata")
    assert artifact_fingerprint(bundle) == original


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_fingerprint_covers_every_required_artifact(bundle, name):
    before = artifact_fingerprint(bundle)
    with (bundle / name).open("ab") as stream:
        stream.write(b" ")
    assert artifact_fingerprint(bundle) != before


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_checkpoint_is_rejected(bundle, monkeypatch, key):
    import healthph_plus.disease_inference as inference

    def load_tokenizer(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return object()

    def load_model(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        return object(), {key: ["classifier.weight"]}

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained", load_model)
    with pytest.raises(ValueError, match="Checkpoint weights"):
        DiseasePredictor(bundle)


def test_environment_override_is_used(bundle, monkeypatch):
    monkeypatch.setenv("HEALTHPH_DISEASE_ARTIFACT_DIR", str(bundle))
    (bundle / "thresholds.json").unlink()
    with pytest.raises(ValueError, match=str(bundle)):
        DiseasePredictor()


def test_tokenizer_vocabulary_must_match_weights(bundle, monkeypatch):
    import healthph_plus.disease_inference as inference

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: ["token"])
    model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(num_embeddings=2))
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *args, **kwargs: (model, {}))
    with pytest.raises(ValueError, match="Tokenizer vocabulary"):
        DiseasePredictor(bundle)


def test_sigmoid_threshold_boundary_and_multiple_labels():
    # Zero logits produce exactly 0.5 and must be included at a 0.5 threshold.
    logits = torch.tensor([[-1., -1., -1., -1.], [0., -1., -1., -1.], [0., 1., -1., 0.]])
    results = results_from_logits(logits, dict.fromkeys(LABELS, 0.5), "test-version")
    assert [r["labels"] for r in results] == [[], ["AURI"], ["AURI", "PN", "COVID"]]
    assert results[1]["probabilities"]["AURI"] == 0.5
    assert results[0]["probabilities"]["AURI"] == pytest.approx(0.2689414)
    assert list(results[0]["probabilities"]) == list(LABELS)


@pytest.mark.parametrize("logits", [torch.zeros(4), torch.zeros(1, 3), torch.full((1, 4), float("nan")), torch.full((1, 4), float("inf"))])
def test_invalid_logits_fail(logits):
    with pytest.raises(RuntimeError, match="invalid disease logits"):
        results_from_logits(logits, dict.fromkeys(LABELS, 0.5), "test-version")


def test_forward_passes_are_serialized_and_validation_precedes_inference():
    predictor = object.__new__(DiseasePredictor)
    predictor._lock = threading.Lock()
    predictor.thresholds = dict.fromkeys(LABELS, 0.5)
    predictor.model_version = "test-version"
    activity = {"active": 0, "maximum": 0, "calls": 0}

    def tokenize(texts, **kwargs):
        return {"count": len(texts)}

    def forward(count):
        activity["calls"] += 1
        activity["active"] += 1
        activity["maximum"] = max(activity["maximum"], activity["active"])
        time.sleep(0.005)
        activity["active"] -= 1
        return SimpleNamespace(logits=torch.zeros(count, 4))

    predictor.tokenizer = tokenize
    predictor.model = forward
    for invalid in [[], ["x"] * 33, ["valid", " "], "text"]:
        with pytest.raises(ValueError):
            predictor.predict_many(invalid)
    assert activity["calls"] == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(predictor.predict, ["text"] * 8))
    assert activity["calls"] == 8
    assert activity["maximum"] == 1


@pytest.mark.model
def test_notebook_parity_batch_order_unicode_and_truncation(real_predictor, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    root = Path(__file__).resolve().parents[1]
    reference_path = root / "tests/fixtures/notebook_inference.py"
    namespace = {
        "model": real_predictor.model, "tokenizer": real_predictor.tokenizer,
        "MAX_LENGTH": MAX_LENGTH, "LABELS": list(LABELS),
        "thresholds": real_predictor.thresholds, "torch": torch,
    }
    exec(compile(reference_path.read_text(), str(reference_path), "exec"), namespace)
    texts = [
        "I have fever, dry cough, and sore throat for 2 days.",
        "ubong may dugo at hirap huminga ako nitong gabi",
        "feeling okay, no symptoms today",
        "  Ubo\n at lagnat — ñ 😷  ",
        "ubo at lagnat " * 500,
        "feeling okay, no symptoms today",
    ]
    assert real_predictor.tokenizer(texts[4], truncation=True, max_length=128, return_tensors="pt")["input_ids"].shape[1] == 128
    start = time.perf_counter()
    batch = real_predictor.predict_many(texts)
    print(f"CPU batch of {len(texts)} texts: {time.perf_counter() - start:.4f}s")
    for text, result in zip(texts, batch):
        reference = namespace["predict_disease_labels"](text)
        single = real_predictor.predict(text)
        assert result["labels"] == single["labels"] == reference["labels"]
        assert result["probabilities"] == pytest.approx(reference["probabilities"], abs=1e-6, rel=0)
        assert result["probabilities"] == pytest.approx(single["probabilities"], abs=1e-6, rel=0)
        assert result["thresholds_used"] == reference["thresholds_used"]
    assert batch[2]["labels"] == batch[5]["labels"]
    assert batch[2]["probabilities"] == pytest.approx(batch[5]["probabilities"], abs=1e-6, rel=0)
