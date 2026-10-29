import json

import pytest

from healthph_xlmr.inference import LABELS, NORMALIZATION, REQUIRED_FILES


def pytest_addoption(parser):
    parser.addoption("--run-model", action="store_true", help="Run tests against the saved XLM-RoBERTa checkpoint")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model"):
        for item in items:
            if "model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --run-model to test the local checkpoint"))


@pytest.fixture(scope="session")
def real_predictor():
    from healthph_xlmr.inference import DiseasePredictor

    # Isolate disease score parity from the keyword response fields.
    return DiseasePredictor(symptom_extractor=FakeSymptomExtractor())


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "source"
    for name in REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "label_order.json").write_text(json.dumps(list(LABELS)))
    (root / "thresholds.json").write_text(json.dumps(dict.fromkeys(LABELS, 0.5)))
    (root / "run_config.json").write_text(json.dumps({
        "run_mode": "full", "smoke_only": False, "label_order": list(LABELS),
        "normalization": NORMALIZATION, "max_length": 128,
    }))
    (root / "best_model/config.json").write_text(json.dumps({
        "model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"],
        "problem_type": "multi_label_classification", "max_position_embeddings": 514,
        "pad_token_id": 1,
        "id2label": {str(i): label for i, label in enumerate(LABELS)},
        "label2id": {label: i for i, label in enumerate(LABELS)},
    }))
    return root


class FakeSymptomExtractor:
    """Explicit disease-only test dependency, never a production fallback."""
    model_version = "test-symptom-version"

    def __init__(self):
        self.calls = []

    def extract_many(self, texts):
        self.calls.append(list(texts))
        return [[] for _ in texts]


@pytest.fixture
def symptom_extractor():
    return FakeSymptomExtractor()
