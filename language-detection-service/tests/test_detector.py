import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from healthph_language import LanguageDetector, normalize_text
from healthph_language import detector as module
from fixtures.notebook_reference import (
    language_keyword_matches, load_language_keywords, normalize_scraped_text,
)

SAMPLES = [
    "I am feeling better today.", "may ubo ako", "sakit ti barukong",
    "BUG-AT   ANG\nDUGHAN!", "uyek and ubo", "subox", "a_ubo", "ubo2",
    "Ceci est une phrase française assez longue pour identifier la langue.",
    "RT @person: [Hello](https://example.com) &amp;amp; goodbye 😷 #health",
    "Look ![photo](https://example.com/x) www.example.com @friend #tag",
    "broken [link](https://example.com", "Hi\u200bthere\x00 friend\r\n\t!",
    "@ubo feeling better", "#ubo feeling better", "&amp;#117;bo today",
    "Naa koy hilanat ug sakit sa lawas.",
]


@pytest.mark.parametrize("text", SAMPLES + ["", "  ", "😷", "12345", "中文"])
def test_cleaner_matches_frozen_notebook(text):
    assert normalize_text(text) == normalize_scraped_text(text)


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models/lid.176.bin").write_bytes(b"test")
    (tmp_path / "models/lid.176.ftz").write_bytes(b"test")
    keywords = tmp_path / "keywords"
    keywords.mkdir()
    (keywords / "hiligaynon_keywords.csv").write_text('\ufeff"ubo"\n"bug-at ang dughan"\n')
    (keywords / "ilocano_keywords.csv").write_text('"uyek"\n')
    model = Mock()
    model.predict.side_effect = lambda texts, k: ([["__label__fr"] for _ in texts], [[0.8] for _ in texts])
    loader = Mock(return_value=model)
    monkeypatch.setattr("fasttext.load_model", loader)
    monkeypatch.setattr(module, "ASSET_ROOT", tmp_path)
    monkeypatch.delenv("HEALTHPH_LANGUAGE_MODEL_PATH", raising=False)
    return tmp_path, model, loader


def test_overrides_boundaries_and_original_text(fake_assets):
    detector = LanguageDetector()
    texts = ["UBO!", "BUG-AT \n ANG\tDUGHAN", "uyek", "uyek and ubo", "subox", "a_ubo", "ubo2", "@ubo hello"]
    results = detector.predict_many(texts)
    assert [r["language"] for r in results] == ["hil", "hil", "ilo", "hil", "fr", "fr", "fr", "hil"]
    assert all(r["fasttext_language"] == "fr" and r["fasttext_confidence"] == 0.8 for r in results)
    assert [r["is_supported"] for r in results] == [True] * 4 + [False] * 3 + [True]
    assert results[0]["prediction_source"] == "keyword_override"
    assert results[4]["prediction_source"] == "fasttext"
    assert results[-1]["cleaned_text"] == "hello"


def test_load_once_chunking_and_single_batch_parity(fake_assets):
    _, model, loader = fake_assets
    detector = LanguageDetector()
    assert detector.predict_many([]) == []
    assert not model.predict.called
    results = detector.predict_many(["hello"] * 65)
    assert [len(call.args[0]) for call in model.predict.call_args_list] == [32, 32, 1]
    assert all(result == detector.predict("hello") for result in results)
    loader.assert_called_once()
    assert detector.predict("hello " * 2000)["language"] == "fr"


@pytest.mark.parametrize("text", [None, 123, True, [], "", " \n ", "😷", "1234 !", "中文", "https://example.com", "@ubo #health"])
def test_invalid_batch_never_calls_model(fake_assets, text):
    detector = LanguageDetector()
    with pytest.raises(ValueError):
        detector.predict_many(["hello", text])
    fake_assets[1].predict.assert_not_called()


def test_model_selection_and_explicit_path(fake_assets, monkeypatch):
    root, _, loader = fake_assets
    assert LanguageDetector().model_path == root / "models/lid.176.bin"
    monkeypatch.setenv("HEALTHPH_LANGUAGE_MODEL_PATH", str(root / "models/lid.176.ftz"))
    assert LanguageDetector().model_path.suffix == ".ftz"
    assert LanguageDetector(model_path=root / "models/lid.176.bin").model_path.suffix == ".bin"
    monkeypatch.delenv("HEALTHPH_LANGUAGE_MODEL_PATH")
    (root / "models/lid.176.bin").unlink()
    assert LanguageDetector().model_path.suffix == ".ftz"
    with pytest.raises(FileNotFoundError):
        LanguageDetector(model_path=root / "missing.bin")
    (root / "models/lid.176.ftz").unlink()
    with pytest.raises(FileNotFoundError, match="Checked"):
        LanguageDetector()


def test_corrupt_model_does_not_fall_back(fake_assets):
    root, _, loader = fake_assets
    loader.side_effect = ValueError("bad model")
    with pytest.raises(RuntimeError, match="Could not load"):
        LanguageDetector()
    loader.assert_called_once_with(str(root / "models/lid.176.bin"))


def test_missing_and_empty_keywords(fake_assets):
    path = fake_assets[0] / "keywords/hiligaynon_keywords.csv"
    path.write_text('\n" "\n')
    with pytest.raises(ValueError, match="empty"):
        LanguageDetector()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        LanguageDetector()


@pytest.mark.model
def test_real_predictions_match_notebook(real_detector):
    cleaned = [normalize_scraped_text(text) for text in SAMPLES]
    labels, scores = real_detector._model.predict(cleaned, k=1)
    hil = language_keyword_matches(pd.Series(SAMPLES), load_language_keywords("hiligaynon_keywords.csv", module.ASSET_ROOT / "keywords"))
    ilo = language_keyword_matches(pd.Series(SAMPLES), load_language_keywords("ilocano_keywords.csv", module.ASSET_ROOT / "keywords"))
    results = real_detector.predict_many(SAMPLES)
    for i, result in enumerate(results):
        expected_model = labels[i][0].replace("__label__", "")
        assert result["language"] == ("hil" if hil[i] else "ilo" if ilo[i] else expected_model)
        assert result["fasttext_language"] == expected_model
        assert result["fasttext_confidence"] == pytest.approx(min(1.0, max(0.0, float(scores[i][0]))))
        assert result["cleaned_text"] == cleaned[i]
        assert real_detector.predict(SAMPLES[i]) == result


@pytest.mark.model
def test_compact_model_and_actual_corruption(tmp_path):
    detector = LanguageDetector(model_path=module.ASSET_ROOT / "models/lid.176.ftz")
    assert detector.predict(SAMPLES[0])["fasttext_language"] == "en"
    bad = tmp_path / "corrupt.bin"
    bad.write_bytes(b"not a FastText model")
    with pytest.raises(RuntimeError, match="Could not load"):
        LanguageDetector(model_path=bad)


@pytest.mark.model
def test_shared_folder_is_independent(tmp_path):
    copied = tmp_path / "handoff"
    shutil.copytree(module.ASSET_ROOT, copied, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache"))
    # Different CWD; only the copied service is on the import path.
    env = {**os.environ, "PYTHONPATH": str(copied)}
    env.pop("HEALTHPH_LANGUAGE_MODEL_PATH", None)
    code = '''import json
from fastapi.testclient import TestClient
from healthph_language.api import create_app
with TestClient(create_app()) as client:
    assert client.get('/health/ready').json()['model'] == 'lid.176.bin'
    response = client.post('/v1/language/predict', json={'text': 'uyek and ubo'})
    assert response.status_code == 200
    print(json.dumps(response.json()))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout)["language"] == "hil"
