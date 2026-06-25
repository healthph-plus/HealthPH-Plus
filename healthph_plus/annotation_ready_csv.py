"""Generate an annotation-ready CSV from merged HealthPH+ post data.

The output follows docs/cleaned_text_annotation_guide.md: annotators see only
cleaned_text and fill disease, misinformation, and sentiment labels.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "processed" / "merged_data.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "training_data" / "annotation_ready.csv"

ANNOTATION_COLUMNS = ["cleaned_text", "disease", "misinformation", "sentiment"]
DEFAULT_DISEASE_VECTOR = "[0, 0, 0, 0]"


@dataclass(frozen=True)
class AnnotationCsvSummary:
    """Summary of a generated annotation CSV."""

    input_csv: Path
    output_csv: Path
    rows_read: int
    rows_written: int
    rows_skipped_empty_text: int


def _label_values(fill_default_labels: bool) -> dict[str, str]:
    if not fill_default_labels:
        return {"disease": "", "misinformation": "", "sentiment": ""}

    return {
        "disease": DEFAULT_DISEASE_VECTOR,
        "misinformation": "0",
        "sentiment": "0",
    }


def _build_output_row(
    source_row: dict[str, str],
    *,
    text_column: str,
    id_column: str,
    include_id: bool,
    labels: dict[str, str],
) -> dict[str, str]:
    output_row = {
        "cleaned_text": source_row[text_column].strip(),
        "disease": labels["disease"],
        "misinformation": labels["misinformation"],
        "sentiment": labels["sentiment"],
    }

    if include_id:
        return {"id": source_row[id_column], **output_row}

    return output_row


def generate_annotation_ready_csv(
    input_csv: Path | str = DEFAULT_INPUT_CSV,
    output_csv: Path | str = DEFAULT_OUTPUT_CSV,
    *,
    text_column: str = "cleaned_text",
    id_column: str = "id",
    include_id: bool = False,
    drop_empty_text: bool = True,
    fill_default_labels: bool = False,
    overwrite: bool = True,
) -> AnnotationCsvSummary:
    """Generate a CSV ready for manual annotation.

    By default, label columns are left blank. Use fill_default_labels=True only
    when an annotation tool requires non-empty placeholders.
    """

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {output_path}. Use overwrite=True to replace it."
        )

    labels = _label_values(fill_default_labels)
    output_columns = ANNOTATION_COLUMNS.copy()
    if include_id:
        output_columns.insert(0, "id")

    rows_read = 0
    rows_written = 0
    rows_skipped_empty_text = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header row: {input_path}")

        missing_columns = [text_column]
        if include_id:
            missing_columns.append(id_column)

        missing_columns = [column for column in missing_columns if column not in reader.fieldnames]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"Input CSV is missing required column(s): {joined}")

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_columns)
            writer.writeheader()

            for source_row in reader:
                rows_read += 1
                cleaned_text = (source_row.get(text_column) or "").strip()
                if drop_empty_text and not cleaned_text:
                    rows_skipped_empty_text += 1
                    continue

                source_row[text_column] = cleaned_text
                writer.writerow(
                    _build_output_row(
                        source_row,
                        text_column=text_column,
                        id_column=id_column,
                        include_id=include_id,
                        labels=labels,
                    )
                )
                rows_written += 1

    return AnnotationCsvSummary(
        input_csv=input_path,
        output_csv=output_path,
        rows_read=rows_read,
        rows_written=rows_written,
        rows_skipped_empty_text=rows_skipped_empty_text,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an annotation-ready CSV from data/processed/merged_data.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Source merged data CSV. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Generated annotation CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--text-column",
        default="cleaned_text",
        help="Column to expose for annotation. Default: cleaned_text",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Source ID column used with --include-id. Default: id",
    )
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="Include the source ID for traceability.",
    )
    parser.add_argument(
        "--keep-empty-text",
        action="store_true",
        help="Keep rows with empty cleaned_text instead of skipping them.",
    )
    parser.add_argument(
        "--fill-default-labels",
        action="store_true",
        help="Fill label cells with [0, 0, 0, 0], 0, and 0 placeholders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = generate_annotation_ready_csv(
        input_csv=args.input,
        output_csv=args.output,
        text_column=args.text_column,
        id_column=args.id_column,
        include_id=args.include_id,
        drop_empty_text=not args.keep_empty_text,
        fill_default_labels=args.fill_default_labels,
        overwrite=args.overwrite,
    )

    print(f"Wrote {summary.rows_written} rows to {summary.output_csv}")
    print(f"Read {summary.rows_read} rows from {summary.input_csv}")
    print(f"Skipped {summary.rows_skipped_empty_text} rows with empty cleaned_text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

