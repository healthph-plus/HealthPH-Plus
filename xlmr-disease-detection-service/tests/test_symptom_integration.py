"""Disease integration with injected extractors and the real keyword default."""

import re
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from healthph_xlmr.inference import DiseasePredictor
from healthph_xlmr.api import create_app


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 0, 2, 1
    is_fast = True
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 4

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        return {"count": len(texts)}


class Model:
    config = SimpleNamespace(bos_token_id=0, eos_token_id=2, pad_token_id=1)

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=4)

    def float(self):
        return self

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, count):
        return SimpleNamespace(logits=torch.zeros(count, 4))


class Extractor:
    model_version = "test-extractor-version"

    def __init__(self):
        self.calls = []
        self.predictor = None

    def extract_many(self, texts):
        assert self.predictor is None or not self.predictor._lock.locked()
        self.calls.append(list(texts))
        return [[{
            "text": match.group(), "normalized_symptom": "cough",
            "start_index": match.start(), "end_index": match.end(),
            "confidence": 0.9, "negated": text[:match.start()].endswith("walang "),
        } for match in re.finditer("ubo", text, flags=re.IGNORECASE)] for text in texts]


@pytest.fixture
def disease_loaders(monkeypatch):
    import healthph_xlmr.inference as inference

    tokenizer = Tokenizer()
    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(inference.AutoModelForSequenceClassification, "from_pretrained",
                        lambda *args, **kwargs: (Model(), {}))
    return tokenizer


def test_extractor_receives_exact_text_and_preserves_disease_outputs(bundle, disease_loaders):
    extractor = Extractor()
    predictor = DiseasePredictor(bundle, symptom_extractor=extractor)
    extractor.predictor = predictor
    texts = [" 😷  walang ubo\n at Ubo  ", "feeling okay", "ubo ubo"]
    batch = predictor.predict_many(texts)
    assert extractor.calls[-1] == texts
    assert [item["text"] for item in batch[0]["symptoms"]] == ["ubo", "Ubo"]
    assert [item["negated"] for item in batch[0]["symptoms"]] == [True, False]
    assert batch[0]["symptoms"][0]["start_index"] == texts[0].index("ubo")
    assert batch[1]["symptoms"] == []
    assert len(batch[2]["symptoms"]) == 2
    for text, result in zip(texts, batch):
        assert result == predictor.predict(text)
        assert result["probabilities"] == dict.fromkeys(("AURI", "PN", "TB", "COVID"), 0.5)
        assert result["symptom_model_version"] == extractor.model_version


def test_extractor_failure_is_not_returned_as_empty_symptoms(bundle, disease_loaders):
    extractor = Extractor()
    predictor = DiseasePredictor(bundle, symptom_extractor=extractor)
    with TestClient(create_app(lambda: predictor)) as client:
        def fail(texts):
            raise RuntimeError("symptom extraction failed")

        extractor.extract_many = fail
        response = client.post("/v1/disease/predict", json={"text": "ubo"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Disease inference failed"}


def test_extractor_wrong_batch_size_fails(bundle, disease_loaders):
    extractor = Extractor()
    predictor = DiseasePredictor(bundle, symptom_extractor=extractor)
    extractor.extract_many = lambda texts: []
    with pytest.raises(RuntimeError, match="Symptom extractor returned the wrong number"):
        predictor.predict("ubo")


def test_default_keywords_work_without_symptom_model(bundle, disease_loaders, monkeypatch):
    monkeypatch.delenv("HEALTHPH_SYMPTOM_KEYWORDS_PATH", raising=False)
    predictor = DiseasePredictor(bundle)
    assert not (bundle / "symptom_model").exists()
    with TestClient(create_app(lambda: predictor)) as client:
        texts = ["😷 Walang UBO pero may lagnat", "okay", "dry cough and cough"]
        batch = client.post("/v1/disease/predict-batch", json={"texts": texts})
        assert batch.status_code == 200
        results = batch.json()["results"]
        assert [s["text"] for s in results[0]["symptoms"]] == ["UBO", "lagnat"]
        assert [s["negated"] for s in results[0]["symptoms"]] == [True, False]
        assert results[1]["symptoms"] == []
        assert [s["text"] for s in results[2]["symptoms"]] == ["dry cough", "cough"]
        for text, result in zip(texts, results):
            assert result == client.post("/v1/disease/predict", json={"text": text}).json()
            assert result["probabilities"] == dict.fromkeys(("AURI", "PN", "TB", "COVID"), 0.5)
            for span in result["symptoms"]:
                assert text[span["start_index"]:span["end_index"]] == span["text"]
                assert span["confidence"] == 1.0


def test_keyword_path_override(bundle, disease_loaders, tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text('disease,symptoms\nRI,custom symptom\n')
    predictor = DiseasePredictor(bundle, symptom_keywords_path=path)
    assert [s["text"] for s in predictor.predict("ubo and custom symptom")["symptoms"]] == ["custom symptom"]


def test_combined_http_contract_contains_mentions_and_both_versions(bundle, disease_loaders):
    predictor = DiseasePredictor(bundle, symptom_extractor=Extractor())
    with TestClient(create_app(lambda: predictor)) as client:
        ready = client.get("/health/ready").json()
        assert ready["symptom_model_version"] == predictor.symptom_model_version
        texts = ["😷 walang ubo", "okay", "ubo ubo"]
        batch = client.post("/v1/disease/predict-batch", json={"texts": texts})
        assert batch.status_code == 200
        for text, result in zip(texts, batch.json()["results"]):
            assert result == client.post("/v1/disease/predict", json={"text": text}).json()
        mention = batch.json()["results"][0]["symptoms"][0]
        assert mention["negated"] is True
        assert texts[0][mention["start_index"]:mention["end_index"]] == mention["text"]
