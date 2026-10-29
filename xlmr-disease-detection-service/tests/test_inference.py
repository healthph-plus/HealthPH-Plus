import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from healthph_xlmr.inference import (
    LABELS, REQUIRED_FILES, DiseasePredictor, artifact_fingerprint, normalize_text,
    results_from_logits, validate_artifacts, validate_text,
)


def update_json(path, **values):
    payload = json.loads(path.read_text())
    payload.update(values)
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("text", [None, 1, True, [], {}, "", " \n\t", "x" * 10_001, " " * 10_000 + "x"])
def test_invalid_text(text):
    with pytest.raises(ValueError):
        normalize_text(text)


def test_normalization_preserves_content_and_input_boundary():
    text = "  Ubo\n at\t lagnat — ñ 😷 #Health https://example.com\u00a0 "
    assert validate_text(text) == text
    assert normalize_text(text) == "Ubo at lagnat — ñ 😷 #Health https://example.com"
    assert normalize_text("x" * 10_000) == "x" * 10_000


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_missing_artifacts_fail_before_loading(bundle, name):
    (bundle / name).unlink()
    with pytest.raises(ValueError, match="Missing XLM-RoBERTa inference artifacts"):
        DiseasePredictor(bundle)


@pytest.mark.parametrize("name", ["run_config.json", "best_model/config.json", "thresholds.json", "label_order.json"])
@pytest.mark.parametrize("payload", ["null", "{", "1", '"invalid"'])
def test_malformed_metadata(bundle, name, payload):
    (bundle / name).write_text(payload)
    with pytest.raises(ValueError):
        validate_artifacts(bundle)


@pytest.mark.parametrize("value", [True, "0.5", None, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_thresholds(bundle, value):
    update_json(bundle / "thresholds.json", TB=value)
    with pytest.raises(ValueError, match="Invalid probability threshold"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("thresholds", [{}, {"AURI": 0.5}, dict.fromkeys((*LABELS, "OTHER"), 0.5), []])
def test_threshold_keys_must_match(bundle, thresholds):
    (bundle / "thresholds.json").write_text(json.dumps(thresholds))
    with pytest.raises(ValueError):
        validate_artifacts(bundle)


def test_label_order_must_match(bundle):
    (bundle / "label_order.json").write_text(json.dumps(list(reversed(LABELS))))
    with pytest.raises(ValueError, match="label order"):
        validate_artifacts(bundle)


@pytest.mark.parametrize("values", [
    {"model_type": "electra"}, {"architectures": ["RobertaForSequenceClassification"]},
    {"problem_type": "single_label_classification"}, {"id2label": {"0": "COVID"}},
    {"label2id": {"COVID": 0}}, {"num_labels": 3}, {"num_labels": True},
    {"max_position_embeddings": 129}, {"max_position_embeddings": None}, {"pad_token_id": -1},
])
def test_inconsistent_model_config(bundle, values):
    update_json(bundle / "best_model/config.json", **values)
    with pytest.raises(ValueError):
        validate_artifacts(bundle)


@pytest.mark.parametrize("values", [
    {"label_order": list(reversed(LABELS))}, {"normalization": "lowercase"},
    {"max_length": True}, {"max_length": 1}, {"max_length": 513}, {"max_length": 128.0},
    {"run_mode": "smoke", "smoke_only": True}, {"run_mode": "full", "smoke_only": True},
    {"run_mode": "smoke", "smoke_only": False}, {"run_mode": None}, {"smoke_only": 0},
])
def test_invalid_run_contract_and_smoke_rejected(bundle, values):
    update_json(bundle / "run_config.json", **values)
    with pytest.raises(ValueError):
        validate_artifacts(bundle)


def test_fingerprint_portability_and_training_metadata(bundle, tmp_path):
    before = artifact_fingerprint(bundle)
    target = tmp_path / "relocated"
    shutil.copytree(bundle, target)
    assert artifact_fingerprint(target) == before
    update_json(bundle / "run_config.json", dataset_path="/different/data.csv", run_id="other", device="mps")
    (bundle / "best_model/training_args.bin").write_bytes(b"training")
    (bundle / "best_model/.DS_Store").write_bytes(b"metadata")
    assert artifact_fingerprint(bundle) == before
    update_json(bundle / "run_config.json", max_length=64)
    assert artifact_fingerprint(bundle) != before


@pytest.mark.parametrize("name", [name for name in REQUIRED_FILES if name != "run_config.json"])
def test_fingerprint_covers_each_inference_asset(bundle, name):
    before = artifact_fingerprint(bundle)
    with (bundle / name).open("ab") as stream:
        stream.write(b" ")
    assert artifact_fingerprint(bundle) != before


def test_fingerprint_covers_optional_tokenizer_assets(bundle):
    before = artifact_fingerprint(bundle)
    path = bundle / "best_model/sentencepiece.bpe.model"
    path.write_bytes(b"tokenizer")
    with_optional = artifact_fingerprint(bundle)
    assert with_optional != before
    path.write_bytes(b"changed tokenizer")
    assert artifact_fingerprint(bundle) != with_optional


class FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    is_fast = True
    model_max_length = 512

    def __len__(self):
        return 4


class FakeModel:
    config = SimpleNamespace(bos_token_id=0, eos_token_id=2, pad_token_id=1)

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=4)

    def float(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def eval(self):
        return self


@pytest.fixture
def loaders(monkeypatch):
    import healthph_xlmr.inference as inference
    tokenizer, model, loading_info = FakeTokenizer(), FakeModel(), {}

    def load_tokenizer(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False, "use_fast": True}
        return tokenizer

    def load_model(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False,
                          "use_safetensors": True, "output_loading_info": True}
        return model, loading_info

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained", load_model)
    return tokenizer, model, loading_info


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_weights_rejected(bundle, loaders, key):
    loaders[2][key] = ["classifier.weight"]
    with pytest.raises(ValueError, match="Checkpoint weights"):
        DiseasePredictor(bundle)


@pytest.mark.parametrize("key,value", [("bos_token_id", 99), ("eos_token_id", 99), ("pad_token_id", 0),
                                      ("is_fast", False), ("model_max_length", 64)])
def test_incompatible_tokenizer_rejected(bundle, loaders, key, value):
    setattr(loaders[0], key, value)
    with pytest.raises(ValueError, match="Tokenizer"):
        DiseasePredictor(bundle)


def test_vocab_mismatch_rejected(bundle, loaders):
    loaders[1].get_input_embeddings = lambda: SimpleNamespace(num_embeddings=5)
    with pytest.raises(ValueError, match="Tokenizer vocabulary"):
        DiseasePredictor(bundle)


def test_offline_cpu_loading_and_override_precedence(bundle, loaders, monkeypatch, tmp_path, symptom_extractor):
    monkeypatch.setenv("HEALTHPH_XLMR_DISEASE_ARTIFACT_DIR", str(bundle))
    assert DiseasePredictor(symptom_extractor=symptom_extractor).artifact_dir == bundle.resolve()
    monkeypatch.setenv("HEALTHPH_XLMR_DISEASE_ARTIFACT_DIR", str(tmp_path / "missing"))
    assert DiseasePredictor(bundle, symptom_extractor=symptom_extractor).max_length == 128
    with pytest.raises(ValueError, match="Missing"):
        DiseasePredictor()


def test_sigmoid_threshold_equality_and_multiple_labels():
    logits = torch.tensor([[-1., -1., -1., -1.], [0., -1., -1., -1.], [0., 1., -1., 0.]])
    results = results_from_logits(logits, dict.fromkeys(LABELS, 0.5), "version")
    assert [r["labels"] for r in results] == [[], ["AURI"], ["AURI", "PN", "COVID"]]
    assert results[1]["probabilities"]["AURI"] == 0.5
    assert results[0]["probabilities"]["AURI"] == pytest.approx(0.2689414)
    assert list(results[0]["probabilities"]) == list(LABELS)
    assert results[0]["thresholds_used"] is not results[1]["thresholds_used"]


def test_thresholds_are_not_rounded_to_float32():
    thresholds = {"AURI": 0.5, "PN": 0.50000000001, "TB": 0.0, "COVID": 1.0}
    result = results_from_logits(torch.zeros(1, 4), thresholds, "version")[0]
    assert result["labels"] == ["AURI", "TB"]
    assert result["thresholds_used"] == thresholds


@pytest.mark.parametrize("logits", [torch.zeros(4), torch.zeros(1, 3),
                                   torch.full((1, 4), float("nan")), torch.full((1, 4), float("inf"))])
def test_invalid_logits_fail(logits):
    with pytest.raises(RuntimeError, match="invalid disease logits"):
        results_from_logits(logits, dict.fromkeys(LABELS, 0.5), "version")


def test_forward_passes_serialized_and_batch_validation_precedes_inference(symptom_extractor):
    predictor = object.__new__(DiseasePredictor)
    predictor._lock = threading.Lock()
    predictor.symptom_extractor = symptom_extractor
    predictor.symptom_model_version = symptom_extractor.model_version
    predictor.max_length = 64
    predictor.thresholds = dict.fromkeys(LABELS, 0.5)
    predictor.model_version = "version"
    activity = {"active": 0, "maximum": 0, "calls": 0}

    def tokenize(texts, **kwargs):
        assert texts == ["Ubo 😷"]
        assert kwargs == {"padding": True, "truncation": True, "max_length": 64, "return_tensors": "pt"}
        return {"count": len(texts)}

    def forward(count):
        activity["calls"] += 1
        activity["active"] += 1
        activity["maximum"] = max(activity["maximum"], activity["active"])
        time.sleep(0.005)
        activity["active"] -= 1
        return SimpleNamespace(logits=torch.zeros(count, 4))

    predictor.tokenizer, predictor.model = tokenize, forward
    for invalid in [[], ["x"] * 33, ["valid", " "], "text", ["valid", 123]]:
        with pytest.raises(ValueError):
            predictor.predict_many(invalid)
    assert activity["calls"] == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(predictor.predict, ["  Ubo\n 😷 "] * 8))
    assert activity["calls"] == 8
    assert activity["maximum"] == 1
    predictor.model = lambda **kwargs: SimpleNamespace(logits=torch.zeros(0, 4))
    with pytest.raises(RuntimeError, match="wrong number"):
        predictor.predict("Ubo 😷")
