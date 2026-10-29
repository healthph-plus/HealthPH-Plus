import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from healthph_xlmr.inference import LABELS, artifact_fingerprint, normalize_text


@pytest.mark.model
def test_notebook_parity_batch_order_unicode_normalization_and_truncation(real_predictor, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    reference = Path(__file__).parent / "fixtures/notebook_reference.py"
    spec = importlib.util.spec_from_file_location("notebook_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = module.load_disease_predictor(real_predictor.artifact_dir, device="cpu")
    texts = [
        "I have a cough and fever.", "May ubo at lagnat ako.",
        "Giubo ko ug gihilantan.", "sakit ti barukong",
        "may sakit ako kag nagakulba gid ko", "Feeling okay today.",
        "  Ubo\n at lagnat — ñ 😷 #Health https://example.com  ",
        "ubo at lagnat " * 400, "May ubo at lagnat ako.",
    ]
    encoded = real_predictor.tokenizer(normalize_text(texts[7]), truncation=True,
                                      max_length=real_predictor.max_length, return_tensors="pt")
    assert encoded["input_ids"].shape[1] == real_predictor.max_length
    batch = real_predictor.predict_many(texts)
    for text, result in zip(texts, batch):
        expected = predict(text)
        single = real_predictor.predict(text)
        assert result["labels"] == expected["labels"] == single["labels"]
        assert result["thresholds_used"] == expected["thresholds_used"] == single["thresholds_used"]
        assert result["probabilities"] == pytest.approx(expected["probabilities"], abs=2e-5, rel=2e-5)
        assert result["probabilities"] == pytest.approx(single["probabilities"], abs=2e-5, rel=2e-5)
        assert list(result["probabilities"]) == list(LABELS)
        assert result["model_version"] == real_predictor.model_version
    assert batch[1] == batch[8]
    assert real_predictor.predict(texts[6]) == real_predictor.predict(normalize_text(texts[6]))
    assert real_predictor.model.training is False
    assert str(next(real_predictor.model.parameters()).device) == "cpu"
    del predict
    gc.collect()


@pytest.mark.model
def test_copied_disease_service_offline_and_restart_update(tmp_path):
    service = Path(__file__).resolve().parents[1]
    target = tmp_path / "standalone"
    shutil.copytree(service, target, ignore=shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", "tests", ".DS_Store", ".xlmr-export-*",
    ))
    original_version = artifact_fingerprint(service / "model")
    program = '''
import gc
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from healthph_xlmr.api import create_app
from healthph_xlmr.inference import DEFAULT_ARTIFACT_DIR, LABELS, DiseasePredictor

# Isolate disease checkpoint portability and threshold changes.
# Keyword extraction is tested separately by the shared symptom test suite.
class FakeSymptomExtractor:
    model_version = "explicit-portability-test-fake"
    def extract_many(self, texts):
        return [[] for _ in texts]

def create_disease_app():
    return create_app(lambda: DiseasePredictor(symptom_extractor=FakeSymptomExtractor()))

assert not Path("model").exists()  # Running from a different working directory.
app = create_disease_app()
with TestClient(app) as client:
    before = client.get("/health/ready").json()["model_version"]
    assert before == sys.argv[1]
    response = client.post("/v1/disease/predict", json={"text": "May ubo at lagnat ako."})
    assert response.status_code == 200
    first = response.json()
    assert first["model_version"] == before
    assert app.state.predictor.artifact_dir == DEFAULT_ARTIFACT_DIR
gc.collect()

threshold_path = DEFAULT_ARTIFACT_DIR / "thresholds.json"
threshold_path.write_text(json.dumps(dict.fromkeys(LABELS, 0.0)))
with TestClient(create_disease_app()) as client:
    after = client.get("/health/ready").json()["model_version"]
    response = client.post("/v1/disease/predict", json={"text": "May ubo at lagnat ako."})
    assert response.status_code == 200
    second = response.json()
    assert after != before
    assert second["thresholds_used"] == dict.fromkeys(LABELS, 0.0)
    assert second["labels"] == list(LABELS)
    assert second["model_version"] == after
    assert second["probabilities"] == first["probabilities"]
print(json.dumps({"portable": True, "restart_update": True}))
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(tmp_path / "empty-hf-cache"), PYTHONPATH=str(target))
    env.pop("HEALTHPH_XLMR_DISEASE_ARTIFACT_DIR", None)
    result = subprocess.run([sys.executable, "-c", program, original_version], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"portable": True, "restart_update": True}
