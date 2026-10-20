import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from healthph_sentiment.inference import (
    NORMALIZATION, REQUIRED_FILES, SentimentPredictor, artifact_fingerprint,
    normalize_text, results_from_logits, validate_artifacts,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "source"
    for name in REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "label_mapping.json").write_text(json.dumps({
        "labels": {"0": "calm", "1": "worried"}, "positive_class": 1,
    }))
    (root / "threshold.json").write_text('{"worried_threshold": 0.5}')
    (root / "run_config.json").write_text(json.dumps({
        "model_key": "mbert", "text_normalization": NORMALIZATION, "max_length": 256,
    }))
    (root / "best_model/config.json").write_text(json.dumps({
        "model_type": "bert", "architectures": ["BertForSequenceClassification"],
        "id2label": {"0": "calm", "1": "worried"}, "label2id": {"calm": 0, "worried": 1},
        "max_position_embeddings": 512,
    }))
    return root


@pytest.mark.parametrize("text", [None, 1, True, [], {}, "", " \n\t", "a" * 10_001])
def test_invalid_text(text):
    with pytest.raises(ValueError):
        normalize_text(text)


def test_normalization_preserves_casing_unicode_and_post_content():
    assert normalize_text("  Ubo\n at lagnat — ñ 😷 #Health https://example.com  ") == (
        "Ubo at lagnat — ñ 😷 #Health https://example.com"
    )
    assert normalize_text("a" * 10_000) == "a" * 10_000


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_missing_artifact_fails_before_loading(bundle, name):
    (bundle / name).unlink()
    with pytest.raises(ValueError, match="Missing sentiment inference artifacts"):
        SentimentPredictor(bundle)


@pytest.mark.parametrize("value", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold(bundle, value):
    (bundle / "threshold.json").write_text(json.dumps({"worried_threshold": value}))
    with pytest.raises(ValueError, match="finite probability"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("value", [True, 0, -1, 513, 256.0, "256", None])
def test_invalid_max_length(bundle, value):
    path = bundle / "run_config.json"
    config = json.loads(path.read_text())
    config["max_length"] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_length"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("file,key,value", [
    ("run_config.json", "model_key", "tfidf_logreg"),
    ("run_config.json", "text_normalization", "lowercase"),
    ("label_mapping.json", "labels", {"0": "worried", "1": "calm"}),
    ("label_mapping.json", "positive_class", True),
    ("best_model/config.json", "model_type", "electra"),
    ("best_model/config.json", "num_labels", 3),
    ("best_model/config.json", "id2label", {"0": "LABEL_0", "1": "LABEL_1"}),
    ("best_model/config.json", "label2id", {"worried": 0, "calm": 1}),
    ("best_model/config.json", "problem_type", "multi_label_classification"),
])
def test_incompatible_contract(bundle, file, key, value):
    path = bundle / file
    config = json.loads(path.read_text())
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_artifacts(bundle)


def test_fingerprint_is_portable_and_ignores_training_metadata(bundle, tmp_path):
    before = artifact_fingerprint(bundle)
    target = tmp_path / "relocated"
    shutil.copytree(bundle, target)
    assert artifact_fingerprint(target) == before
    config = json.loads((target / "run_config.json").read_text())
    config.update(dataset_path="/another/machine/data.csv", run_id="another-run")
    (target / "run_config.json").write_text(json.dumps(config))
    (target / "best_model/training_args.bin").write_bytes(b"ignored")
    assert artifact_fingerprint(target) == before
    config["max_length"] = 128
    (target / "run_config.json").write_text(json.dumps(config))
    assert artifact_fingerprint(target) != before


@pytest.mark.parametrize("name", [name for name in REQUIRED_FILES if name != "run_config.json"])
def test_fingerprint_covers_inference_artifacts(bundle, name):
    before = artifact_fingerprint(bundle)
    with (bundle / name).open("ab") as stream:
        stream.write(b" ")
    assert artifact_fingerprint(bundle) != before


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_weights_rejected(bundle, monkeypatch, key):
    import healthph_sentiment.inference as inference

    def tokenizer(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return object()

    def model(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        return object(), {key: ["classifier.weight"]}

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained", model)
    with pytest.raises(ValueError, match="Checkpoint weights"):
        SentimentPredictor(bundle)


def test_tokenizer_mismatch_rejected(bundle, monkeypatch):
    import healthph_sentiment.inference as inference

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: ["one"])
    model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(num_embeddings=2))
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *args, **kwargs: (model, {}))
    with pytest.raises(ValueError, match="Tokenizer vocabulary"):
        SentimentPredictor(bundle)


def test_environment_override(bundle, monkeypatch):
    monkeypatch.setenv("HEALTHPH_SENTIMENT_ARTIFACT_DIR", str(bundle))
    (bundle / "threshold.json").unlink()
    with pytest.raises(ValueError, match=str(bundle)):
        SentimentPredictor()


def test_softmax_threshold_boundary():
    results = results_from_logits(torch.tensor([[0., 0.], [2., 0.], [0., 2.]]), 0.5, "version")
    assert [result["predicted_label_name"] for result in results] == ["worried", "calm", "worried"]
    assert results[0]["worried_probability"] == 0.5
    assert results[1]["worried_probability"] == pytest.approx(0.1192029)


@pytest.mark.parametrize("logits", [torch.zeros(2), torch.zeros(1, 3),
                                   torch.full((1, 2), float("nan")), torch.full((1, 2), float("inf"))])
def test_invalid_logits(logits):
    with pytest.raises(RuntimeError, match="invalid sentiment logits"):
        results_from_logits(logits, 0.5, "version")


def test_validation_precedes_inference_and_forward_passes_are_serialized():
    predictor = object.__new__(SentimentPredictor)
    predictor._lock = threading.Lock()
    predictor.threshold, predictor.max_length, predictor.model_version = 0.5, 256, "version"
    activity = {"active": 0, "maximum": 0, "calls": 0}

    def tokenize(texts, **kwargs):
        assert texts == ["Ubo 😷"]
        assert kwargs["max_length"] == 256
        return {"count": len(texts)}

    def forward(count):
        activity["calls"] += 1
        activity["active"] += 1
        activity["maximum"] = max(activity["maximum"], activity["active"])
        time.sleep(0.005)
        activity["active"] -= 1
        return SimpleNamespace(logits=torch.zeros(count, 2))

    predictor.tokenizer, predictor.model = tokenize, forward
    for invalid in [[], ["x"] * 33, ["valid", " "], "text"]:
        with pytest.raises(ValueError):
            predictor.predict_many(invalid)
    assert activity["calls"] == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(predictor.predict, ["  Ubo\n😷  "] * 8))
    assert activity["calls"] == 8
    assert activity["maximum"] == 1
