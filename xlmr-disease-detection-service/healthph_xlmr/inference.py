"""Offline CPU inference for the notebook's XLM-RoBERTa disease classifier."""

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

from healthph_symptoms.keywords import KeywordSymptomExtractor, SymptomExtractor

LABELS = ("AURI", "PN", "TB", "COVID")
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
    "label_order.json",
    "thresholds.json",
)
OPTIONAL_TOKENIZER_FILES = (
    "sentencepiece.bpe.model", "special_tokens_map.json", "added_tokens.json",
)


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
        raise ValueError(f"Missing XLM-RoBERTa inference artifacts in {root}: {', '.join(missing)}")

    def read_json(name, expected_type=dict):
        with (root / name).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, expected_type):
            raise ValueError(f"{name} must contain a JSON {expected_type.__name__}")
        return payload

    labels = read_json("label_order.json", list)
    run = read_json("run_config.json")
    if labels != list(LABELS) or run.get("label_order") != list(LABELS):
        raise ValueError(f"Saved label order must be {list(LABELS)}")
    if run.get("normalization") != NORMALIZATION:
        raise ValueError("Unsupported saved text normalization")
    if run.get("run_mode") not in ("full", "smoke") or type(run.get("smoke_only")) is not bool:
        raise ValueError("Missing or invalid run-mode metadata")
    if run["smoke_only"] != (run["run_mode"] == "smoke"):
        raise ValueError("Inconsistent run-mode metadata")
    if run["smoke_only"]:
        raise ValueError("Smoke bundles cannot be served; export a completed full run")

    config = read_json("best_model/config.json")
    if (config.get("model_type") != "xlm-roberta"
            or config.get("architectures") != ["XLMRobertaForSequenceClassification"]
            or config.get("problem_type") != "multi_label_classification"
            or config.get("id2label") != {str(i): label for i, label in enumerate(LABELS)}
            or config.get("label2id") != {label: i for i, label in enumerate(LABELS)}
            or type(config.get("num_labels", 4)) is not int
            or config.get("num_labels", 4) != 4):
        raise ValueError("Model configuration does not match the four-label XLM-RoBERTa contract")
    max_length = run.get("max_length")
    max_positions = config.get("max_position_embeddings")
    padding_idx = config.get("pad_token_id")
    # XLM-R position IDs start at padding_idx + 1 for non-padding tokens.
    if (type(max_length) is not int or not 2 <= max_length <= 512
            or type(max_positions) is not int or type(padding_idx) is not int
            or padding_idx < 0 or max_length > max_positions - padding_idx - 1):
        raise ValueError("Invalid saved token limit or model position capacity")

    thresholds = read_json("thresholds.json")
    if set(thresholds) != set(LABELS):
        raise ValueError("thresholds.json must contain exactly AURI, PN, TB, COVID")
    for label, value in thresholds.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid probability threshold for {label}")
    return {
        "max_length": max_length,
        "thresholds": {label: float(thresholds[label]) for label in LABELS},
    }


def inference_files(root: Path) -> list[str]:
    """The same explicit asset list is used for export and versioning."""
    return sorted([*REQUIRED_FILES, *(
        f"best_model/{name}" for name in OPTIONAL_TOKENIZER_FILES
        if (root / "best_model" / name).is_file()
    )])


def artifact_fingerprint(root: Path) -> str:
    """Hash inference content, independent of location and training metadata."""
    settings = validate_artifacts(root)
    digest = hashlib.sha256(b"healthph-xlmr-disease-inference-v1\0")
    digest.update(json.dumps({**settings, "normalization": NORMALIZATION}, sort_keys=True).encode())
    for name in inference_files(root):
        if name == "run_config.json":
            continue  # Effective settings are included above; training paths are metadata.
        file_digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(name.encode() + b"\0" + file_digest.digest())
    return "sha256:" + digest.hexdigest()


def results_from_logits(
    logits: torch.Tensor, thresholds: dict[str, float], model_version: str,
) -> list[dict]:
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not torch.isfinite(logits).all():
        raise RuntimeError("Model returned invalid disease logits")
    probabilities = torch.sigmoid(logits.float()).cpu().tolist()
    return [{
        # Compare the emitted scores to the saved thresholds without rounding
        # thresholds to float32. This matches the notebook's float64 thresholds.
        "labels": [label for label, score in zip(LABELS, scores) if score >= thresholds[label]],
        "probabilities": dict(zip(LABELS, scores)),
        "thresholds_used": thresholds.copy(),
        "model_version": model_version,
    } for scores in probabilities]


class DiseasePredictor:
    """One local CPU model; reuse it across requests. Forward passes are serialized."""

    def __init__(
        self, artifact_dir: Path | str | None = None, *,
        symptom_extractor: SymptomExtractor | None = None,
        symptom_keywords_path: Path | str | None = None,
    ):
        self.artifact_dir = Path(
            artifact_dir if artifact_dir is not None else
            os.environ.get("HEALTHPH_XLMR_DISEASE_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
        ).expanduser().resolve()
        settings = validate_artifacts(self.artifact_dir)
        self.max_length = settings["max_length"]
        self.thresholds = settings["thresholds"]
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
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
        )):
            raise ValueError(f"Checkpoint weights do not match model configuration: {loading_info}")
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer vocabulary size does not match model embeddings")
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(self.tokenizer, key) != getattr(self.model.config, key):
                raise ValueError(f"Tokenizer {key} does not match model configuration")
        if not self.tokenizer.is_fast or self.max_length > self.tokenizer.model_max_length:
            raise ValueError("Tokenizer must support fast encoding and the saved token limit")
        self.model.float().to("cpu").eval()
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
        normalized = [normalize_text(text) for text in texts]
        with self._lock, torch.inference_mode():
            encoded = self.tokenizer(
                normalized, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            logits = self.model(**encoded).logits
            if logits.ndim != 2 or logits.shape[0] != len(texts):
                raise RuntimeError("Model returned the wrong number of predictions")
            results = results_from_logits(logits, self.thresholds, self.model_version)
        # Preserve offsets into the exact text before disease normalization.
        symptoms = self.symptom_extractor.extract_many(texts)
        if len(symptoms) != len(texts):
            raise RuntimeError("Symptom extractor returned the wrong number of predictions")
        for result, mentions in zip(results, symptoms):
            result.update(symptoms=mentions, symptom_model_version=self.symptom_model_version)
        return results
