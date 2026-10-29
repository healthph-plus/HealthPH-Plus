"""Literal symptom keyword matching using only the Python standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Protocol

PACKAGE_DIR = Path(__file__).resolve().parent
MASTER_KEYWORDS = PACKAGE_DIR.parent / "docs/keywords/keywords_master.csv"
DEFAULT_KEYWORDS = PACKAGE_DIR / "keywords_master.csv"
if not DEFAULT_KEYWORDS.is_file():
    DEFAULT_KEYWORDS = MASTER_KEYWORDS
MAX_TEXT_CHARACTERS = 10_000
MAX_BATCH_SIZE = 32
# These source-list entries name diseases or senses, rather than symptoms.
EXCLUDED_KEYWORDS = frozenset({"covid", "covid-19", "tb", "tuberculosis", "pneumonia", "smell", "taste"})
MATCHER_REVISION = "literal-keywords-v2:longest-disjoint:codepoint:negation-eight-original-words"

_BOUNDARY = re.compile(r"[.!?;:\n]|\b(?:but|however|although|except|pero|ngunit|subalit|kaso|apan|ngem|nguni|apang)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|don['’]t|doesn['’]t|didn['’]t|without|denies|denied|negative for|walang|wala|hindi|di|walay|dili|wa|awan|saan|diak|indi)\b", re.I)
_PRESENCE = re.compile(r"\b(?:may|meron|mayroon|naa|naay|adda)\b|\b(?:and|but|at|ug|kag|ken)\s+(?:has|have|having)\b", re.I)
_NOT_ONLY = re.compile(r"\b(?:not only|hindi lang|hindi lamang|di lang|indi lang|dili lang)\b", re.I)


def normalize_keyword(text: str) -> str:
    return " ".join(text.split()).casefold()


def _is_negated(text: str, start: int, completed_spans: list[dict] | None = None) -> bool:
    """Apply bounded literal cue rules; this does not infer clinical context."""
    prefix = text[:start]
    boundaries = list(_BOUNDARY.finditer(prefix))
    offset = boundaries[-1].end() if boundaries else 0
    clause = prefix[offset:]
    words = list(re.finditer(r"\S+", clause))
    if len(words) > 8:
        offset += words[-8].start()
        clause = prefix[offset:]
    # Negation inside a matched symptom ("walang panlasa") belongs to that
    # phrase and must not negate the next symptom.
    for span in reversed(completed_spans or []):
        if span["end_index"] <= offset:
            break
        left, right = max(0, span["start_index"] - offset), span["end_index"] - offset
        fragment = _NEGATION.sub(lambda match: " " * len(match.group()), clause[left:right])
        clause = clause[:left] + fragment + clause[right:]
    clause = _NOT_ONLY.sub("", clause)
    negations = list(_NEGATION.finditer(clause))
    return bool(negations and not _PRESENCE.search(clause[negations[-1].end():]))


class SymptomExtractor(Protocol):
    model_version: str

    def extract_many(self, texts: list[str]) -> list[list[dict]]: ...


class KeywordSymptomExtractor:
    """Match master-list phrases without tokenizers, model weights, or training.

    Aliases supply optional English names only; they do not add search terms.
    Confidence 1.0 denotes a keyword hit, not a clinical/model probability.
    """

    def __init__(self, keywords_path: Path | str | None = None, *,
                 aliases_path: Path | str | None = None):
        self.keywords_path = Path(keywords_path if keywords_path is not None else
                                  os.environ.get("HEALTHPH_SYMPTOM_KEYWORDS_PATH", DEFAULT_KEYWORDS)).expanduser().resolve()
        with self.keywords_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"disease", "symptoms"}.issubset(reader.fieldnames):
                raise ValueError("Keyword CSV must contain disease and symptoms columns")
            terms = set()
            for row in reader:
                if not row.get("disease") or not row.get("symptoms") or None in row:
                    raise ValueError("Each keyword row must contain a disease and symptoms")
                # Commas separate terms; slashes in the source list separate
                # spelling alternatives ("danggason/gi danggas ko").
                terms.update(normalize_keyword(term) for term in re.split(r"[,/]", row["symptoms"]))
        terms.difference_update(EXCLUDED_KEYWORDS | {""})
        if not terms:
            raise ValueError("Keyword CSV must contain symptom keywords")

        vocabulary = json.loads(Path(aliases_path if aliases_path is not None else
                                          PACKAGE_DIR / "aliases.json").read_text(encoding="utf-8"))
        if not isinstance(vocabulary, dict) or not isinstance(vocabulary.get("aliases"), list):
            raise ValueError("aliases.json must contain an aliases list")
        canonical = {}
        for alias in vocabulary["aliases"]:
            if (not isinstance(alias, dict) or not isinstance(alias.get("text"), str)
                    or not alias["text"].strip() or not isinstance(alias.get("normalized_symptom"), str)
                    or not alias["normalized_symptom"].strip()):
                raise ValueError("Invalid symptom alias")
            key, name = normalize_keyword(alias["text"]), alias["normalized_symptom"]
            if key in canonical and canonical[key] != name:
                raise ValueError(f"Conflicting normalization for keyword {key!r}")
            canonical[key] = name
        self._patterns = []
        entries = []
        for term in sorted(terms):
            phrase = r"\s+".join(re.escape(part) for part in term.split())
            name = canonical.get(term)
            self._patterns.append((re.compile(r"(?<!\w)" + phrase + r"(?!\w)", re.I), name))
            entries.append((term, name))
        settings = {"revision": MATCHER_REVISION, "entries": entries,
                    "rules": [rule.pattern for rule in (_BOUNDARY, _NEGATION, _PRESENCE, _NOT_ONLY)]}
        self.model_version = "sha256:" + hashlib.sha256(
            json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def extract(self, text: str) -> list[dict]:
        return self.extract_many([text])[0]

    def extract_many(self, texts: list[str]) -> list[list[dict]]:
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
            raise ValueError(f"texts must be a list containing 1–{MAX_BATCH_SIZE} texts")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if len(text) > MAX_TEXT_CHARACTERS:
                raise ValueError(f"text must contain at most {MAX_TEXT_CHARACTERS} characters")
        return [self._extract(text) for text in texts]

    def _extract(self, text: str) -> list[dict]:
        candidates = [
            {"text": match.group(), "normalized_symptom": name,
             "start_index": match.start(), "end_index": match.end(),
             "confidence": 1.0, "negated": False}
            for pattern, name in self._patterns for match in pattern.finditer(text)
        ]
        # Prefer complete phrases over shorter embedded keywords; retain every
        # distinct occurrence and return disjoint spans in original-text order.
        candidates.sort(key=lambda item: (-(item["end_index"] - item["start_index"]), item["start_index"]))
        selected = []
        occupied = bytearray(len(text))
        for candidate in candidates:
            start, end = candidate["start_index"], candidate["end_index"]
            if not any(occupied[start:end]):
                selected.append(candidate)
                occupied[start:end] = b"\x01" * (end - start)
        selected.sort(key=lambda item: item["start_index"])
        for index, span in enumerate(selected):
            span["negated"] = _is_negated(text, span["start_index"], selected[:index])
        return selected


