"""Validate annotations and prepare reproducible, disjoint multilabel splits."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
from pathlib import Path

LABELS = ("AURI", "PN", "TB", "COVID")
NORMALIZATION = "collapse whitespace and strip; preserve casing and Unicode"
MAX_LENGTH = 128


def parse_labels(value: str) -> list[int]:
    try:
        labels = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise ValueError("annotate must be a list of four binary integers") from exc
    if (not isinstance(labels, list) or len(labels) != len(LABELS)
            or any(type(item) is not int or item not in (0, 1) for item in labels)):
        raise ValueError("annotate must contain four integer zeros/ones in AURI, PN, TB, COVID order")
    return labels


def load_annotations(paths, architecture, conflict_policy="reject"):
    """Deduplicate whitespace variants; reject or exclude entire conflicting groups."""
    if architecture not in {"electra", "xlm-roberta"}:
        raise ValueError("Unsupported disease architecture")
    if conflict_policy not in {"reject", "exclude"}:
        raise ValueError("Conflict policy must be reject or exclude")
    records, fingerprints, sources, conflicts = [], {}, [], set()
    raw_rows = 0
    for path in paths:
        path = Path(path).expanduser().resolve()
        payload = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        if not reader.fieldnames or not {"post", "annotate"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: CSV requires post and annotate columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate column names")
        for source_row, row in enumerate(reader, 2):
            raw_rows += 1
            location = {"file_index": len(sources) - 1, "csv_row": source_row}
            try:
                text = row["post"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("post must be a nonblank string")
                labels = parse_labels(row["annotate"])
            except ValueError as exc:
                raise ValueError(f"{path}: row {source_row}: {exc}") from exc
            normalized = " ".join(text.split())
            fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if fingerprint in fingerprints:
                previous = records[fingerprints[fingerprint]]
                if previous["labels"] != labels:
                    if conflict_policy == "reject":
                        raise ValueError(
                            f"{path}: row {source_row}: conflicting annotations for duplicate post. "
                            "Correct the annotations or use --conflict-policy exclude to exclude the entire post group."
                        )
                    conflicts.add(fingerprint)
                previous["source_rows"].append(location)
                continue
            fingerprints[fingerprint] = len(records)
            records.append({
                "text": normalized if architecture == "xlm-roberta" else text,
                "labels": labels, "text_fingerprint": fingerprint, "source_rows": [location],
            })
    excluded = [{"text_fingerprint": row["text_fingerprint"], "source_rows": row["source_rows"]}
                for row in records if row["text_fingerprint"] in conflicts]
    records = [row for row in records if row["text_fingerprint"] not in conflicts]
    if not records:
        raise ValueError("Dataset is empty or has no consistently annotated posts")
    return records, {
        "sources": sources, "raw_rows": raw_rows, "retained_posts": len(records),
        "duplicate_rows_removed": sum(len(row["source_rows"]) - 1 for row in records),
        "conflicting_posts": len(conflicts), "conflicting_rows_excluded": sum(len(row["source_rows"]) for row in excluded),
        "excluded_conflicts": excluded,
        "retained_posts_over_service_character_limit": sum(len(row["text"]) > 10_000 for row in records),
        "duplicate_identity": NORMALIZATION, "conflict_policy": conflict_policy,
    }


def split_annotations(records, validation_size=0.1, test_size=0.1, seed=42):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    if not (0 < validation_size < 1 and 0 < test_size < 1 and validation_size + test_size < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    if len(records) < 3:
        raise ValueError("Need enough distinct posts for nonempty train, validation, and test splits")
    labels = np.asarray([record["labels"] for record in records])
    try:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval, test = next(splitter.split(np.zeros(len(records)), labels))
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=validation_size / (1 - test_size), random_state=seed,
        )
        train, validation = next(splitter.split(np.zeros(len(trainval)), labels[trainval]))
    except ValueError as exc:
        raise ValueError("Dataset is too small for the requested split fractions") from exc
    indices = {"train": trainval[train], "validation": trainval[validation], "test": test}
    parts = {name: [records[int(index)] for index in selected] for name, selected in indices.items()}
    for name, rows in parts.items():
        counts = np.asarray([row["labels"] for row in rows]).reshape(-1, 4).sum(axis=0)
        missing = [label for label, count in zip(LABELS, counts) if count == 0 or count == len(rows)]
        if not rows or missing:
            raise ValueError(
                f"{name}: every disease needs positive and negative examples; insufficient support for {missing}. "
                "Add annotated posts or adjust --validation-size, --test-size, or --seed."
            )
    return parts
