from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from healthph_language.api import create_app


class FakeDetector:
    model_path = Path("lid.176.bin")

    def __init__(self):
        self.calls = []
        self.fail = False

    def predict(self, text):
        return self.predict_many([text])[0]

    def predict_many(self, texts):
        self.calls.append(texts)
        if self.fail:
            raise RuntimeError("internal failure")
        return [{"language": "en", "cleaned_text": text.lower(), "is_supported": True,
                 "prediction_source": "fasttext", "fasttext_language": "en",
                 "fasttext_confidence": 0.9} for text in texts]


@pytest.fixture
def service():
    detector = FakeDetector()
    app = create_app(lambda: detector)
    with TestClient(app) as client:
        detector.calls.clear()
        yield client, detector
    assert app.state.detector is None


def test_health_and_ordered_raw_requests(service):
    client, detector = service
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready", "model": "lid.176.bin"}
    response = client.post("/v1/language/predict", json={"text": "@ubo Hello"})
    assert response.status_code == 200
    assert detector.calls == [["@ubo Hello"]]
    texts = ["HELLO", "GOODBYE", "HELLO"]
    response = client.post("/v1/language/predict-batch", json={"texts": texts})
    assert response.status_code == 200
    assert [r["cleaned_text"] for r in response.json()["results"]] == [t.lower() for t in texts]


@pytest.mark.parametrize("body", [{}, {"text": None}, {"text": 42}, {"text": " "}, {"text": "😷"},
    {"text": "123"}, {"text": "https://example.com"}, {"text": "hello", "extra": 1}, {"text": "a" * 10001}])
def test_invalid_single(service, body):
    client, detector = service
    assert client.post("/v1/language/predict", json=body).status_code == 422
    assert detector.calls == []


@pytest.mark.parametrize("texts", [[], ["hello", " "], ["a"] * 33, [None], "hello", ["a" * 10001]])
def test_invalid_batch_is_atomic(service, texts):
    client, detector = service
    assert client.post("/v1/language/predict-batch", json={"texts": texts}).status_code == 422
    assert detector.calls == []


def test_limit_boundaries_and_inference_error(service):
    client, detector = service
    assert client.post("/v1/language/predict-batch", json={"texts": ["a" * 10000] * 32}).status_code == 200
    detector.fail = True
    response = client.post("/v1/language/predict", json={"text": "hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Language inference failed"}


def test_not_ready():
    with TestClient(create_app(FakeDetector)) as client:
        client.app.state.detector = None
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/language/predict", json={"text": "hello"}).status_code == 503


def test_startup_failure():
    def missing():
        raise FileNotFoundError("missing model")
    with pytest.raises(FileNotFoundError, match="missing model"):
        with TestClient(create_app(missing)):
            pass


@pytest.mark.model
def test_real_api(real_detector):
    with TestClient(create_app(lambda: real_detector)) as client:
        texts = ["may ubo ako", "sakit ti barukong", "I am feeling better today."]
        response = client.post("/v1/language/predict-batch", json={"texts": texts})
        assert response.status_code == 200
        assert response.json()["results"] == real_detector.predict_many(texts)
