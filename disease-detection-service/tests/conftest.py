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

    # Missing artifacts must fail an explicitly requested model test, not skip it.
    return DiseasePredictor()
