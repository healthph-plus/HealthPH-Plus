"""FastText inference with the notebook's curated language overrides."""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from threading import Lock
from typing import Literal, TypedDict

from .cleaning import normalize_text

ASSET_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGUAGES = frozenset({"en", "tl", "ceb", "fil", "ilo", "hil"})
MAX_BATCH_SIZE = 32
MAX_TEXT_CHARACTERS = 10_000


class Prediction(TypedDict):
    language: str
    cleaned_text: str
    is_supported: bool
    prediction_source: Literal["fasttext", "keyword_override"]
    fasttext_language: str
    fasttext_confidence: float


def clean_valid_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    cleaned = normalize_text(text)
    if not re.search(r"[a-zA-Z]", cleaned):
        raise ValueError("text must contain an ASCII letter after cleaning")
    return cleaned


def load_keyword_pattern(path: Path) -> re.Pattern:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        keywords = {row[0].strip() for row in csv.reader(handle) if row and row[0].strip()}
    if not keywords:
        raise ValueError(f"Language keyword list is empty: {path}")
    alternatives = [re.escape(word).replace(r"\ ", r"\s+")
                    for word in sorted(keywords, key=len, reverse=True)]
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


class LanguageDetector:
    """Load assets once and reuse this instance for single or bulk predictions.

    Python bulk calls accept any number of texts, processing 32 at a time.
    HTTP request size limits are enforced by the API, not offline notebooks.
    """

    def __init__(self, model_path: str | Path | None = None,
                 keyword_dir: str | Path | None = None):
        explicit = model_path if model_path is not None else os.getenv("HEALTHPH_LANGUAGE_MODEL_PATH")
        if explicit is not None:
            selected = Path(explicit).expanduser().resolve()
        else:
            candidates = [ASSET_ROOT / "models" / name for name in ("lid.176.bin", "lid.176.ftz")]
            selected = next((path for path in candidates if path.is_file()), None)
            if selected is None:
                raise FileNotFoundError("FastText model missing. Checked: " + ", ".join(map(str, candidates)))
        if not selected.is_file():
            raise FileNotFoundError(f"FastText model missing: {selected}")
        self.model_path = selected
        keyword_dir = Path(keyword_dir) if keyword_dir is not None else ASSET_ROOT / "keywords"
        self._patterns = [
            (language, load_keyword_pattern(keyword_dir / filename))
            for language, filename in (("hil", "hiligaynon_keywords.csv"), ("ilo", "ilocano_keywords.csv"))
        ]
        import fasttext

        try:
            self._model = fasttext.load_model(str(selected))
        except Exception as exc:
            raise RuntimeError(f"Could not load FastText model: {selected}") from exc
        self._lock = Lock()

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        # Validate the whole input before inference; never return a partial batch.
        cleaned = [clean_valid_text(text) for text in texts]
        results: list[Prediction] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = cleaned[start:start + MAX_BATCH_SIZE]
            with self._lock:
                # Always use list input, including single predictions. This also
                # avoids FastText 0.9.3's scalar conversion issue with NumPy 2.
                labels, scores = self._model.predict(batch, k=1)
            if len(labels) != len(batch) or len(scores) != len(batch):
                raise RuntimeError("FastText returned an incomplete batch")
            for original, normalized, label, score in zip(
                texts[start:start + MAX_BATCH_SIZE], batch, labels, scores, strict=True
            ):
                fasttext_language = label[0].removeprefix("__label__")
                confidence = float(score[0])
                if not math.isfinite(confidence):
                    raise RuntimeError("FastText returned a non-finite confidence")
                language = fasttext_language
                source = "fasttext"
                for override, pattern in self._patterns:
                    if pattern.search(original):
                        language, source = override, "keyword_override"
                        break
                results.append({
                    "language": language,
                    "cleaned_text": normalized,
                    "is_supported": language in SUPPORTED_LANGUAGES,
                    "prediction_source": source,
                    "fasttext_language": fasttext_language,
                    # FastText can slightly exceed 1 due to numerical rounding.
                    "fasttext_confidence": min(1.0, max(0.0, confidence)),
                })
        return results
