from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from healthph_language.api import create_app


class FakeDetector:
    model_path = Path("lid.176.bin")

    def __init__(self):
        self.calls = []
        self.options = []
        self.fail = False

    def predict(self, text, **options):
        return self.predict_many([text], **options)[0]

    def predict_many(self, texts, *, top_k=1, secondary_min_confidence=0.30):
        self.calls.append(texts)
        self.options.append({"top_k": top_k, "secondary_min_confidence": secondary_min_confidence})
        if self.fail:
            raise RuntimeError("internal failure")
        results = [{"language": "en", "cleaned_text": text.lower(), "is_supported": True,
                    "prediction_source": "fasttext", "fasttext_language": "en",
                    "fasttext_confidence": 0.6} for text in texts]
        if top_k == 2:
            for result in results:
                result["fasttext_predictions"] = [{"language": "en", "confidence": 0.6}]
                if secondary_min_confidence <= 0.3:
                    result["fasttext_predictions"].append({"language": "tl", "confidence": 0.3})
        return results


@pytest.fixture
def service():
    detector = FakeDetector()
    app = create_app(lambda: detector)
    with TestClient(app) as client:
        detector.calls.clear()
        detector.options.clear()
        yield client, detector
    assert app.state.detector is None


def test_health_and_ordered_raw_requests(service):
    client, detector = service
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready", "model": "lid.176.bin"}
    response = client.post("/v1/language/predict", json={"text": "@ubo Hello"})
    assert response.status_code == 200
    assert response.json() == {
        "language": "en", "cleaned_text": "@ubo hello", "is_supported": True,
        "prediction_source": "fasttext", "fasttext_language": "en", "fasttext_confidence": 0.6,
    }
    assert detector.calls == [["@ubo Hello"]]
    texts = ["HELLO", "GOODBYE", "HELLO"]
    response = client.post("/v1/language/predict-batch", json={"texts": texts})
    assert response.status_code == 200
    assert [r["cleaned_text"] for r in response.json()["results"]] == [t.lower() for t in texts]
    assert all("fasttext_predictions" not in result for result in response.json()["results"])


@pytest.mark.parametrize("options,count", [
    ({"top_k": 2}, 2), ({"top_k": 2, "secondary_min_confidence": 0.3}, 2),
    ({"top_k": 2, "secondary_min_confidence": 0.31}, 1),
    ({"top_k": 2, "secondary_min_confidence": 0}, 2),
    ({"top_k": 1, "secondary_min_confidence": 1}, 0),
])
def test_prediction_options_forwarded_for_single_and_ordered_batch(service, options, count):
    client, detector = service
    single = client.post("/v1/language/predict", json={"text": "HELLO", **options})
    batch = client.post("/v1/language/predict-batch", json={"texts": ["HELLO", "BYE", "HELLO"], **options})
    assert single.status_code == batch.status_code == 200
    results = batch.json()["results"]
    assert results[0] == results[2] == single.json()
    assert [result["cleaned_text"] for result in results] == ["hello", "bye", "hello"]
    if count:
        expected = [{"language": "en", "confidence": 0.6}, {"language": "tl", "confidence": 0.3}]
        assert all(result["fasttext_predictions"] == expected[:count] for result in results)
    else:
        assert all("fasttext_predictions" not in result for result in results)
    assert detector.options == [{"top_k": 1, "secondary_min_confidence": 0.30} | options] * 2


@pytest.mark.parametrize("endpoint,payload", [
    ("predict", {"text": "hello"}), ("predict-batch", {"texts": ["hello"]}),
])
@pytest.mark.parametrize("options", [
    {"top_k": value} for value in (0, 3, 1.0, True, None, "2", [])
] + [
    {"secondary_min_confidence": value} for value in (-0.01, 1.01, True, None, "0.3", [])
])
def test_invalid_options_reject_request_before_inference(service, endpoint, payload, options):
    client, detector = service
    response = client.post(f"/v1/language/{endpoint}", json=payload | options)
    assert response.status_code == 422
    assert detector.calls == []


@pytest.mark.parametrize("nonfinite", ["NaN", "Infinity", "-Infinity", "1e400"])
@pytest.mark.parametrize("endpoint,text_field", [
    ("predict", '"text":"hello"'), ("predict-batch", '"texts":["hello"]'),
])
def test_nonfinite_threshold_rejects_request(service, endpoint, text_field, nonfinite):
    client, detector = service
    response = client.post(
        f"/v1/language/{endpoint}",
        content='{' + text_field + ',"secondary_min_confidence":' + nonfinite + '}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert detector.calls == []


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
@pytest.mark.parametrize("options", [{}, {"top_k": 2}, {"top_k": 2, "secondary_min_confidence": 0}])
def test_real_api(real_detector, options):
    with TestClient(create_app(lambda: real_detector)) as client:
        texts = ["may ubo ako", "sakit ti barukong", "I am feeling better today.",
                 "Hello po, can I ask kung saan ang clinic?"]
        response = client.post("/v1/language/predict-batch", json={"texts": texts, **options})
        assert response.status_code == 200
        assert response.json()["results"] == real_detector.predict_many(texts, **options)
        single = client.post("/v1/language/predict", json={"text": texts[-1], **options})
        assert single.status_code == 200
        assert single.json() == response.json()["results"][-1]
