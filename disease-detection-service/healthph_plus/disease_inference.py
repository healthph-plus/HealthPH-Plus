"""Offline CPU inference for the saved HealthPH+ ELECTRA disease classifier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from healthph_symptoms.keywords import KeywordSymptomExtractor, SymptomExtractor

LABELS = ("AURI", "PN", "TB", "COVID")
MAX_LENGTH = 128
MAX_TEXT_CHARACTERS = 10_000
MAX_BATCH_SIZE = 32
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model"
REQUIRED_FILES = (
    "best_model/model.safetensors",
    "best_model/config.json",
    "best_model/tokenizer.json",
    "best_model/tokenizer_config.json",
    "label_order.json",
    "thresholds.json",
)


def validate_text(text: str) -> str:
    """Validate without altering text supplied by the caller."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"text must contain at most {MAX_TEXT_CHARACTERS} characters")
    return text


def validate_artifacts(root: Path) -> dict[str, float]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing disease inference artifacts in {root}: {', '.join(missing)}")

    def read_json(name):
        with (root / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    if read_json("label_order.json") != list(LABELS):
        raise ValueError(f"label_order.json must contain {list(LABELS)} in that order")
    config = read_json("best_model/config.json")
    expected_id2label = {str(i): label for i, label in enumerate(LABELS)}
    expected_label2id = {label: i for i, label in enumerate(LABELS)}
    if (
        config.get("model_type") != "electra"
        or config.get("problem_type") != "multi_label_classification"
        or config.get("id2label") != expected_id2label
        or config.get("label2id") != expected_label2id
        or config.get("num_labels", len(LABELS)) != len(LABELS)
    ):
        raise ValueError("Model configuration does not match the four-label ELECTRA contract")

    thresholds = read_json("thresholds.json")
    if not isinstance(thresholds, dict) or set(thresholds) != set(LABELS):
        raise ValueError("thresholds.json must contain exactly AURI, PN, TB, COVID")
    for label, value in thresholds.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid probability threshold for {label}")
    return {label: float(thresholds[label]) for label in LABELS}


def artifact_fingerprint(root: Path) -> str:
    """Hash content and relative names; moving an unchanged bundle keeps its ID."""
    names = {"label_order.json", "thresholds.json"}
    names.update(
        path.relative_to(root).as_posix()
        for path in (root / "best_model").rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
    digest = hashlib.sha256(b"healthph-electra-inference-v1:max_length=128\0")
    for name in sorted(names):
        file_digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(name.encode("utf-8") + b"\0" + file_digest.digest())
    return "sha256:" + digest.hexdigest()


def results_from_logits(
    logits: torch.Tensor, thresholds: dict[str, float], model_version: str
) -> list[dict]:
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not torch.isfinite(logits).all():
        raise RuntimeError("Model returned invalid disease logits")
    probabilities = torch.sigmoid(logits).cpu()
    selected = probabilities >= torch.tensor(
        [thresholds[label] for label in LABELS], dtype=probabilities.dtype
    )
    return [
        {
            "labels": [label for label, active in zip(LABELS, flags) if active],
            "probabilities": dict(zip(LABELS, scores)),
            "thresholds_used": thresholds.copy(),
            "model_version": model_version,
        }
        for scores, flags in zip(probabilities.tolist(), selected.tolist())
    ]


class DiseasePredictor:
    """One local model instance; the lock serializes all forward passes."""

    def __init__(
        self, artifact_dir: Path | str | None = None, *,
        symptom_extractor: SymptomExtractor | None = None,
        symptom_keywords_path: Path | str | None = None,
    ):
        self.artifact_dir = Path(
            artifact_dir if artifact_dir is not None
            else os.environ.get("HEALTHPH_DISEASE_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
        ).expanduser().resolve()
        self.thresholds = validate_artifacts(self.artifact_dir)
        self.model_version = artifact_fingerprint(self.artifact_dir)
        model_dir = self.artifact_dir / "best_model"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
        )
        if any(loading_info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        )):
            raise ValueError(f"Checkpoint weights do not match model configuration: {loading_info}")
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer vocabulary size does not match the model embeddings")
        self.model.to("cpu").eval()
        self._lock = threading.Lock()
        self.symptom_extractor = (
            symptom_extractor if symptom_extractor is not None else
            KeywordSymptomExtractor(symptom_keywords_path)
        )
        self.symptom_model_version = self.symptom_extractor.model_version

    def predict(self, text: str) -> dict:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[dict]:
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError(f"texts must be a list containing 1–{MAX_BATCH_SIZE} texts")
        for text in texts:
            validate_text(text)
        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(
                texts, padding=True, truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = self.model(**encoded).logits
            if logits.shape[0] != len(texts):
                raise RuntimeError("Model returned the wrong number of predictions")
            results = results_from_logits(logits, self.thresholds, self.model_version)
        # Match the complete submitted text, independently of disease truncation.
        symptoms = self.symptom_extractor.extract_many(texts)
        if len(symptoms) != len(texts):
            raise RuntimeError("Symptom extractor returned the wrong number of predictions")
        for result, mentions in zip(results, symptoms):
            result.update(symptoms=mentions, symptom_model_version=self.symptom_model_version)
        return results
