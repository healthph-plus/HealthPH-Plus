import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.model
def test_notebook_parity_batch_order_unicode_and_truncation(real_predictor, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reference = Path(__file__).parent / "fixtures/notebook_reference.py"
    spec = importlib.util.spec_from_file_location("notebook_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = module.load_sentiment_predictor(real_predictor.artifact_dir, device="cpu")
    texts = [
        "I am worried that my fever and cough are getting worse.",
        "Buti na lang okay na siya ngayon.",
        "Kinakabahan ako, my fever is not going away.",
        "sakit ti barukong", "may sakit ako kag nagakulba gid ko",
        "  Ubo\n at lagnat — ñ 😷 #Health  ",
        "worried about my fever " * 400,
        "Buti na lang okay na siya ngayon.",
    ]
    encoded = real_predictor.tokenizer(texts[6], truncation=True,
                                      max_length=real_predictor.max_length, return_tensors="pt")
    assert encoded["input_ids"].shape[1] == real_predictor.max_length
    batch = real_predictor.predict_many(texts)
    for text, result in zip(texts, batch):
        expected = predict(text)
        single = real_predictor.predict(text)
        assert result["predicted_label"] == expected["predicted_label"] == single["predicted_label"]
        assert result["predicted_label_name"] == expected["predicted_label_name"]
        assert result["threshold_used"] == expected["threshold_used"]
        assert result["worried_probability"] == pytest.approx(expected["worried_probability"], abs=1e-6, rel=0)
        assert result["worried_probability"] == pytest.approx(single["worried_probability"], abs=1e-6, rel=0)
    assert batch[1] == batch[7]


@pytest.mark.model
def test_copied_service_is_offline_portable_and_restart_activates_update(tmp_path):
    service = Path(__file__).resolve().parents[1]
    target = tmp_path / "standalone"
    shutil.copytree(service, target, ignore=shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", "tests", ".DS_Store",
    ))
    program = '''
import json
from pathlib import Path
from fastapi.testclient import TestClient
from healthph_sentiment.api import create_app

with TestClient(create_app()) as client:
    before = client.get("/health/ready").json()["model_version"]
    first = client.post("/v1/sentiment/predict", json={"text": "I am worried about my fever."}).json()
    assert first["model_version"] == before

threshold_path = Path("model/threshold.json")
threshold = json.loads(threshold_path.read_text())["worried_threshold"]
new_threshold = 1.0 if threshold != 1.0 else 0.0
threshold_path.write_text(json.dumps({"worried_threshold": new_threshold}))

with TestClient(create_app()) as client:
    after = client.get("/health/ready").json()["model_version"]
    second = client.post("/v1/sentiment/predict", json={"text": "I am worried about my fever."}).json()
    assert after != before
    assert second["threshold_used"] == new_threshold
    assert second["model_version"] == after
print(json.dumps({"portable": True, "restart_update": True}))
'''
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HEALTHPH_SENTIMENT_ARTIFACT_DIR=str(target / "model"), PYTHONPATH=str(target))
    result = subprocess.run([sys.executable, "-c", program], cwd=target, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"portable": True, "restart_update": True}
