"""Verify literal extraction and offsets without symptom model dependencies."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from healthph_symptoms.keywords import KeywordSymptomExtractor


@pytest.fixture(scope="module")
def extractor():
    return KeywordSymptomExtractor()


@pytest.mark.parametrize("text,phrases", [
    ("😷 May UBO at lagnat ako.", ["UBO", "lagnat"]),
    ("dry cough, cough, coughé, scough, coughing", ["dry cough", "cough"]),
    ("hilanat ug sakit akong ulo", ["hilanat", "sakit akong ulo"]),
    ("Narigrigat ti anges", ["Narigrigat ti anges"]),
    ("Ginakapoy", ["Ginakapoy"]),
    ("😷 sore\n  throat", ["sore\n  throat"]),
    ("ubo, UBO, ubo", ["ubo", "UBO", "ubo"]),
    ("danggason at gi danggas ko", ["danggason", "gi danggas ko"]),
    ("COVID, tb, tuberculosis, pneumonia, smell, taste", []),
    ("lovely weather today", []),
])
def test_master_keywords_and_exact_offsets(extractor, text, phrases):
    spans = extractor.extract(text)
    assert [span["text"] for span in spans] == phrases
    previous_end = 0
    for span in spans:
        assert previous_end <= span["start_index"] < span["end_index"]
        assert text[span["start_index"]:span["end_index"]] == span["text"]
        assert span["confidence"] == 1.0
        previous_end = span["end_index"]


@pytest.mark.parametrize("text,flags", [
    ("walang ubo at sipon", [True, True]),
    ("walang ubo pero may lagnat", [True, False]),
    ("walang ubo at may lagnat", [True, False]),
    ("no cough; fever today", [True, False]),
    ("no cough or fever", [True, True]),
    ("not only cough but colds", [False, False]),
    ("walang panlasa at lagnat", [False, False]),
    ("no loss of taste", [True]),
    ("dili makatilaw ug hilanat", [False, False]),
    ("hindi nawawala ang ubo", [False]),
])
def test_simple_negation(extractor, text, flags):
    assert [span["negated"] for span in extractor.extract(text)] == flags


def write_keywords(path, terms):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["disease", "symptoms"])
        writer.writerow(["RI", terms])


def test_only_csv_terms_match_and_unknown_normalizations_remain_null(tmp_path):
    path = tmp_path / "keywords.csv"
    write_keywords(path, "ubo,custom symptom")
    extractor = KeywordSymptomExtractor(path)
    spans = extractor.extract("fever and ubo and custom symptom")
    assert [span["text"] for span in spans] == ["ubo", "custom symptom"]
    assert [span["normalized_symptom"] for span in spans] == ["cough", None]


def test_versions_depend_on_effective_keywords_and_normalizations(tmp_path):
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    write_keywords(first, "ubo,lagnat,ubo")
    write_keywords(second, "LAGNAT,ubo")
    baseline = KeywordSymptomExtractor(first).model_version
    assert KeywordSymptomExtractor(second).model_version == baseline
    write_keywords(second, "ubo,lagnat,sipon")
    assert KeywordSymptomExtractor(second).model_version != baseline
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"aliases": [{"text": "ubo", "normalized_symptom": "custom"}]}))
    assert KeywordSymptomExtractor(first, aliases_path=aliases).model_version != baseline


def test_full_input_batch_order_and_duplicates(extractor):
    long_text = "okay " * 1800 + "😷 UBO"
    texts = [long_text, "ubo at lagnat", "feeling okay", "ubo at lagnat"]
    results = extractor.extract_many(texts)
    assert results == [extractor.extract(text) for text in texts]
    assert results[0][0]["start_index"] == long_text.index("UBO")
    assert results[2] == []


@pytest.mark.parametrize("texts", [[], "ubo", [""], [" "], [None], ["ubo", 1], ["x" * 10001], ["ubo"] * 33])
def test_invalid_input(extractor, texts):
    with pytest.raises(ValueError):
        extractor.extract_many(texts)


@pytest.mark.parametrize("contents", ["keyword\nubo\n", "disease,symptoms\nRI,\n", "disease,symptoms\nRI,tb\n"])
def test_invalid_keyword_list_fails(tmp_path, contents):
    path = tmp_path / "invalid.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        KeywordSymptomExtractor(path)


def test_keyword_path_override_precedes_environment(tmp_path, monkeypatch):
    explicit, environment = tmp_path / "explicit.csv", tmp_path / "environment.csv"
    write_keywords(explicit, "ubo")
    write_keywords(environment, "lagnat")
    monkeypatch.setenv("HEALTHPH_SYMPTOM_KEYWORDS_PATH", str(environment))
    assert KeywordSymptomExtractor().extract("ubo at lagnat")[0]["text"] == "lagnat"
    assert KeywordSymptomExtractor(explicit).extract("ubo at lagnat")[0]["text"] == "ubo"


def test_extractor_imports_and_runs_without_nlp_packages():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys; sys.path.insert(0,sys.argv[1]); "
        "sys.modules.update(torch=None,transformers=None,spacy=None,nltk=None); "
        "from healthph_symptoms.keywords import KeywordSymptomExtractor; "
        "assert KeywordSymptomExtractor().extract('ubo')[0]['text']=='ubo'"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
