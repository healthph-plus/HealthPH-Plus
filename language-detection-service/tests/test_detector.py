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
    model.predict.side_effect = lambda texts, k: (
        [["__label__fr", "__label__tl"][:k] for _ in texts],
        [[0.8, 0.15][:k] for _ in texts],
    )
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
    assert all(r["fasttext_language"] == "fr" and r["fasttext_confidence"] == 0.8
               and r["fasttext_confidence_percentage"] == 80.0 for r in results)
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


def test_default_response_includes_confidence_percentage(fake_assets):
    result = LanguageDetector().predict("HELLO")
    assert result == {
        "language": "fr", "cleaned_text": "hello", "is_supported": False,
        "prediction_source": "fasttext", "fasttext_language": "fr",
        "fasttext_confidence": 0.8, "fasttext_confidence_percentage": 80.0,
    }
    fake_assets[1].predict.assert_called_once_with(["hello"], k=1)


@pytest.mark.parametrize("top_score,second_score,threshold,count", [
    (0.6, 0.2999, 0.30, 1), (0.6, 0.30, 0.30, 2), (0.6, 0.3001, 0.30, 2),
    (0.2, 0.1, 0.30, 1), (0.2, 0.1, 0.1, 2), (0.6, 0.3, 1, 1),
    (1.00001, -0.00001, 0, 2), (0.654321, 0.299999, 0.30, 1),
])
def test_secondary_threshold_and_clamping(fake_assets, top_score, second_score, threshold, count):
    model = fake_assets[1]
    model.predict.side_effect = lambda texts, k: (
        [["__label__en", "__label__tl"][:k] for _ in texts],
        [[top_score, second_score][:k] for _ in texts],
    )
    result = LanguageDetector().predict("hello", top_k=2, secondary_min_confidence=threshold)
    expected = [
        {"language": "en", "confidence": min(1.0, max(0.0, top_score)),
         "confidence_percentage": round(min(1.0, max(0.0, top_score)) * 100, 2)},
        {"language": "tl", "confidence": min(1.0, max(0.0, second_score)),
         "confidence_percentage": round(min(1.0, max(0.0, second_score)) * 100, 2)},
    ]
    assert result["fasttext_predictions"] == expected[:count]
    assert result["fasttext_confidence"] == expected[0]["confidence"]
    assert result["fasttext_confidence_percentage"] == expected[0]["confidence_percentage"]
    model.predict.assert_called_once_with(["hello"], k=2)


def test_two_language_default_threshold_preserves_keyword_overrides(fake_assets):
    fake_assets[1].predict.side_effect = lambda texts, k: (
        [["__label__tl", "__label__en"][:k] for _ in texts],
        [[0.6, 0.3][:k] for _ in texts],
    )
    detector = LanguageDetector()
    result = detector.predict("ubo and uyek", top_k=2)
    assert result.pop("fasttext_predictions") == [
        {"language": "tl", "confidence": 0.6, "confidence_percentage": 60.0},
        {"language": "en", "confidence": 0.3, "confidence_percentage": 30.0},
    ]
    assert result == detector.predict("ubo and uyek")
    assert result["language"] == "hil"
    assert result["prediction_source"] == "keyword_override"
    assert result["fasttext_language"] == "tl"
    assert result["fasttext_confidence_percentage"] == 60.0


def test_two_language_chunking_order_and_single_batch_parity(fake_assets):
    detector = LanguageDetector()
    assert detector.predict_many([], top_k=2) == []
    fake_assets[1].predict.assert_not_called()
    texts = ["HELLO", "GOODBYE"] * 33
    results = detector.predict_many(texts, top_k=2, secondary_min_confidence=0.1)
    assert [len(call.args[0]) for call in fake_assets[1].predict.call_args_list] == [32, 32, 2]
    assert all(call.kwargs == {"k": 2} for call in fake_assets[1].predict.call_args_list)
    assert [result["cleaned_text"] for result in results] == [text.lower() for text in texts]
    for text, result in zip(texts[:2], results[:2]):
        assert result == detector.predict(text, top_k=2, secondary_min_confidence=0.1)
        assert result["fasttext_predictions"] == [
            {"language": "fr", "confidence": 0.8, "confidence_percentage": 80.0},
            {"language": "tl", "confidence": 0.15, "confidence_percentage": 15.0},
        ]
        assert result["is_supported"] is False


@pytest.mark.parametrize("options", [
    {"top_k": value} for value in (0, 3, 1.0, True, None, "2", [])
] + [
    {"secondary_min_confidence": value}
    for value in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True, None, "0.3", [])
])
def test_invalid_prediction_options_never_call_model(fake_assets, options):
    detector = LanguageDetector()
    with pytest.raises(ValueError):
        detector.predict("hello", **options)
    with pytest.raises(ValueError):
        detector.predict_many([], **options)
    fake_assets[1].predict.assert_not_called()


@pytest.mark.parametrize("top_k,index", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_candidates_are_rejected_even_below_threshold(fake_assets, top_k, index, nonfinite):
    scores = [0.6, 0.3]
    scores[index] = nonfinite
    fake_assets[1].predict.side_effect = lambda texts, k: (
        [["__label__en", "__label__tl"][:k] for _ in texts], [scores[:k] for _ in texts],
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        LanguageDetector().predict("hello", top_k=top_k, secondary_min_confidence=1)


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([["__label__en"]], [[0.6, 0.3]]),
    ([["__label__en", "__label__tl"]], [[0.6]]),
])
def test_incomplete_model_predictions_fail(fake_assets, labels, scores):
    fake_assets[1].predict.side_effect = None
    fake_assets[1].predict.return_value = labels, scores
    with pytest.raises(RuntimeError, match="incomplete"):
        LanguageDetector().predict("hello", top_k=2)


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
        assert result["fasttext_confidence_percentage"] == round(result["fasttext_confidence"] * 100, 2)
        assert result["cleaned_text"] == cleaned[i]
        assert real_detector.predict(SAMPLES[i]) == result


@pytest.mark.model
@pytest.mark.parametrize("model_name", ["lid.176.bin", "lid.176.ftz"])
def test_real_two_language_predictions(model_name):
    detector = LanguageDetector(model_path=module.ASSET_ROOT / "models" / model_name)
    texts = SAMPLES + ["Hello po, can I ask kung saan ang clinic?"]
    labels, scores = detector._model.predict([normalize_text(text) for text in texts], k=2)
    defaults = detector.predict_many(texts)
    results = detector.predict_many(texts, top_k=2)
    unfiltered = detector.predict_many(texts, top_k=2, secondary_min_confidence=0)
    for text, original, result, both, label, score in zip(
        texts, defaults, results, unfiltered, labels, scores, strict=True,
    ):
        expected = [
            {"language": code.removeprefix("__label__"),
             "confidence": min(1.0, max(0.0, float(confidence))),
             "confidence_percentage": round(min(1.0, max(0.0, float(confidence))) * 100, 2)}
            for code, confidence in zip(label, score, strict=True)
        ]
        assert both["fasttext_predictions"] == expected
        assert result["fasttext_predictions"] == expected[:1] + [
            candidate for candidate in expected[1:] if candidate["confidence"] >= 0.30
        ]
        assert result == detector.predict(text, top_k=2)
        assert {key: value for key, value in result.items() if key != "fasttext_predictions"} == original


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
