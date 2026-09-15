import pytest


def pytest_addoption(parser):
    parser.addoption("--run-model", action="store_true", help="Test the bundled FastText models")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model"):
        for item in items:
            if "model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --run-model to test bundled models"))


@pytest.fixture(scope="session")
def real_detector():
    from healthph_language import LanguageDetector

    return LanguageDetector()
