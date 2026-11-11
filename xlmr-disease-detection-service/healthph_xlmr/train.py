"""Run optional XLM-R disease training without importing the inference server."""

from pathlib import Path

from healthph_disease_training.runner import main


if __name__ == "__main__":
    main("xlm-roberta", Path(__file__).resolve().parents[1])
