"""Offline CPU inference for the notebook's binary mBERT sentiment model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "calm", 1: "worried"}
NORMALIZATION = "collapse whitespace and strip; preserve casing and Unicode"
MAX_TEXT_CHARACTERS = 10_000
MAX_BATCH_SIZE = 32
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model"
REQUIRED_FILES = (
    "best_model/model.safetensors",
    "best_model/config.json",
    "best_model/tokenizer.json",
    "best_model/tokenizer_config.json",
    "run_config.json",
    "label_mapping.json",
    "threshold.json",
)
OPTIONAL_TOKENIZER_FILES = ("vocab.txt", "special_tokens_map.json", "added_tokens.json")


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError(f"text must contain at most {MAX_TEXT_CHARACTERS} characters")
    return text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", validate_text(text)).strip()


def validate_artifacts(root: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing sentiment inference artifacts in {root}: {', '.join(missing)}")

    def read_json(name):
        with (root / name).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return payload

    mapping = read_json("label_mapping.json")
    expected_labels = {str(key): value for key, value in LABELS.items()}
    if (mapping.get("labels") != expected_labels
            or type(mapping.get("positive_class")) is not int
            or mapping["positive_class"] != 1):
        raise ValueError("Label mapping must contain 0 = calm, 1 = worried, positive_class = 1")
    run = read_json("run_config.json")
    if run.get("model_key") != "mbert" or run.get("text_normalization") != NORMALIZATION:
        raise ValueError("Unsupported saved model type or normalization contract")
    config = read_json("best_model/config.json")
    if (config.get("model_type") != "bert"
            or config.get("architectures") != ["BertForSequenceClassification"]
            or config.get("id2label") != expected_labels
            or config.get("label2id") != {value: key for key, value in LABELS.items()}
            or config.get("num_labels", 2) != 2
            or config.get("problem_type") not in (None, "single_label_classification")):
        raise ValueError("Model configuration does not match the binary mBERT contract")
    max_length = run.get("max_length")
    max_positions = config.get("max_position_embeddings")
    if (type(max_positions) is not int or max_positions < 1
            or type(max_length) is not int or not 1 <= max_length <= max_positions):
        raise ValueError("max_length must be a positive integer within model position capacity")
    threshold = read_json("threshold.json").get("worried_threshold")
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("worried_threshold must be a finite probability in [0, 1]")
    return {"max_length": max_length, "worried_threshold": float(threshold)}


def artifact_fingerprint(root: Path) -> str:
    """Identify inference content, independent of paths and training metadata."""
    settings = validate_artifacts(root)
    names = set(REQUIRED_FILES) - {"run_config.json"}
    names.update(
        f"best_model/{name}" for name in OPTIONAL_TOKENIZER_FILES
        if (root / "best_model" / name).is_file()
    )
    digest = hashlib.sha256(b"healthph-mbert-sentiment-inference-v1\0")
    digest.update(json.dumps({**settings, "normalization": NORMALIZATION}, sort_keys=True).encode())
    for name in sorted(names):
        file_digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(name.encode() + b"\0" + file_digest.digest())
    return "sha256:" + digest.hexdigest()


def results_from_logits(logits: torch.Tensor, threshold: float, model_version: str) -> list[dict]:
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise RuntimeError("Model returned invalid sentiment logits")
    probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
    results = []
    for probability in probabilities:
        label = int(probability >= threshold)
        results.append({
            "predicted_label": label,
            "predicted_label_name": LABELS[label],
            "worried_probability": probability,
            "threshold_used": threshold,
            "model_version": model_version,
        })
    return results


class SentimentPredictor:
    """One model instance; reuse it across requests. Inputs are validated atomically."""

    def __init__(self, artifact_dir: Path | str | None = None):
        self.artifact_dir = Path(
            artifact_dir if artifact_dir is not None else
            os.environ.get("HEALTHPH_SENTIMENT_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
        ).expanduser().resolve()
        settings = validate_artifacts(self.artifact_dir)
        self.max_length = settings["max_length"]
        self.threshold = settings["worried_threshold"]
        self.model_version = artifact_fingerprint(self.artifact_dir)
        model_dir = self.artifact_dir / "best_model"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading_info.get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
        )):
            raise ValueError(f"Checkpoint weights do not match model configuration: {loading_info}")
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer vocabulary size does not match model embeddings")
        self.model.to("cpu").eval()
        self._lock = threading.Lock()

    def predict(self, text: str) -> dict:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[dict]:
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError(f"texts must be a list containing 1–{MAX_BATCH_SIZE} texts")
        normalized = [normalize_text(text) for text in texts]
        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(
                normalized, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self.model(**encoded).logits
            if logits.ndim != 2 or logits.shape[0] != len(texts):
                raise RuntimeError("Model returned the wrong number of predictions")
            return results_from_logits(logits, self.threshold, self.model_version)
