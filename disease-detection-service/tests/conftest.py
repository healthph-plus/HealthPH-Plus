import pytest


def pytest_addoption(parser):
    parser.addoption("--run-model", action="store_true", help="Run tests against the saved ELECTRA checkpoint")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model"):
        for item in items:
            if "model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --run-model to test the local checkpoint"))


@pytest.fixture(scope="session")
def real_predictor():
    from healthph_plus.disease_inference import DiseasePredictor

    # Isolate disease score parity from the keyword response fields.
    return DiseasePredictor(symptom_extractor=FakeSymptomExtractor())


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
