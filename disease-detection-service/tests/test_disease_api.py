import pytest
from fastapi.testclient import TestClient

from healthph_plus.disease_api import create_app
from healthph_plus.disease_inference import LABELS


class FakePredictor:
    model_version = "test-version"

    def __init__(self):
        self.calls = []

    def predict(self, text):
        return self.predict_many([text])[0]

    def predict_many(self, texts):
        self.calls.append(texts)
        return [{
            "labels": [], "probabilities": dict.fromkeys(LABELS, 0.1),
            "thresholds_used": dict.fromkeys(LABELS, 0.5), "model_version": self.model_version,
        } for text in texts]


@pytest.fixture
def service():
    predictor = FakePredictor()
    with TestClient(create_app(lambda: predictor)) as client:
        yield client, predictor


def test_startup_loads_once_and_health_does_not_run_inference():
    created = []

    def factory():
        predictor = FakePredictor()
        created.append(predictor)
        return predictor

    app = create_app(factory)
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/health/live").json() == {"status": "ok"}
            assert client.get("/health/ready").json() == {"status": "ready", "model_version": "test-version"}
        assert len(created) == 1
        assert len(created[0].calls) == 1
    assert app.state.predictor is None


def test_not_ready_without_startup():
    client = TestClient(create_app())
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.post("/v1/disease/predict", json={"text": "ubo"}).status_code == 503


@pytest.mark.parametrize("failure", ["load", "warmup"])
def test_startup_failure_propagates(failure):
    def factory():
        if failure == "load":
            raise ValueError("invalid artifacts")
        predictor = FakePredictor()

        def fail(text):
            raise ValueError("invalid forward pass")

        predictor.predict = fail
        return predictor

    app = create_app(factory)
    with pytest.raises(ValueError):
        with TestClient(app):
            pass
    assert app.state.predictor is None


def test_api_preserves_text_and_returns_full_contract(service):
    client, predictor = service
    text = "  Ubo\n at lagnat 😷  "
    response = client.post("/v1/disease/predict", json={"text": text})
    assert response.status_code == 200
    assert set(response.json()) == {"labels", "probabilities", "thresholds_used", "model_version"}
    assert predictor.calls[-1] == [text]
    texts = ["second", "first", "second"]
    response = client.post("/v1/disease/predict-batch", json={"texts": texts})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3
    assert predictor.calls[-1] == texts


@pytest.mark.parametrize("body", [{}, {"text": None}, {"text": 123}, {"text": True}, {"text": []}, {"text": ""}, {"text": " \n"}, {"text": "x" * 10_001}, {"text": "ubo", "threshold": 0.9}])
def test_invalid_single_does_not_run_inference(service, body):
    client, predictor = service
    before = len(predictor.calls)
    assert client.post("/v1/disease/predict", json=body).status_code == 422
    assert len(predictor.calls) == before


@pytest.mark.parametrize("body", [{}, {"texts": []}, {"texts": ["ubo"] * 33}, {"texts": "ubo"}, {"texts": ["valid", " "]}, {"texts": ["valid", None]}, {"texts": ["valid", 1]}, {"texts": ["x" * 10_001]}])
def test_invalid_batch_is_atomic(service, body):
    client, predictor = service
    before = len(predictor.calls)
    assert client.post("/v1/disease/predict-batch", json=body).status_code == 422
    assert len(predictor.calls) == before


def test_request_boundaries_and_malformed_json(service):
    client, _ = service
    assert client.post("/v1/disease/predict", json={"text": "x" * 10_000}).status_code == 200
    assert client.post("/v1/disease/predict-batch", json={"texts": ["ubo"] * 32}).status_code == 200
    assert client.post("/v1/disease/predict", content="{", headers={"Content-Type": "application/json"}).status_code == 422


def test_inference_failure_is_an_error_not_empty_labels(service):
    client, predictor = service

    def fail(texts):
        raise RuntimeError("internal implementation detail")

    predictor.predict_many = fail
    response = client.post("/v1/disease/predict", json={"text": "ubo"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Disease inference failed"}


@pytest.mark.model
def test_real_model_api(real_predictor):
    with TestClient(create_app(lambda: real_predictor)) as client:
        assert client.get("/health/ready").json()["model_version"] == real_predictor.model_version
        texts = ["ubo at lagnat", "feeling okay, no symptoms today", "ubo at lagnat"]
        batch = client.post("/v1/disease/predict-batch", json={"texts": texts})
        assert batch.status_code == 200
        for text, result in zip(texts, batch.json()["results"]):
            single = client.post("/v1/disease/predict", json={"text": text}).json()
            assert result["labels"] == single["labels"]
            assert result["probabilities"] == pytest.approx(single["probabilities"], abs=1e-6, rel=0)
