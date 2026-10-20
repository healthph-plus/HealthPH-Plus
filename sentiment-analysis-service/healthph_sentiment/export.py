"""Prepare a validated inference-only bundle from a completed notebook export."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from .inference import (
    DEFAULT_ARTIFACT_DIR, OPTIONAL_TOKENIZER_FILES, REQUIRED_FILES,
    SentimentPredictor, validate_artifacts,
)


def export_model_bundle(source_dir: Path | str, target_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> Path:
    """Stage, load, and warm up before replacement. Stop the service before updating."""
    source = Path(source_dir).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and target bundle directories must not overlap")
    validate_artifacts(source)
    if target.exists() and not target.is_dir():
        raise ValueError("Target must be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sentiment-export-", dir=target.parent) as temporary:
        stage = Path(temporary) / "bundle"
        names = list(REQUIRED_FILES) + [
            f"best_model/{name}" for name in OPTIONAL_TOKENIZER_FILES
            if (source / "best_model" / name).is_file()
        ]
        for name in names:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        predictor = SentimentPredictor(stage)
        predictor.predict("sentiment inference export check")
        del predictor
        backup = Path(temporary) / "previous"
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except BaseException:
            if backup.exists():
                os.replace(backup, target)
            raise
    return target


def main():
    parser = argparse.ArgumentParser(description="Export a completed notebook mBERT bundle")
    parser.add_argument("source_dir", help="Saved run's mbert directory, not its checkpoints")
    parser.add_argument("--target-dir", default=str(DEFAULT_ARTIFACT_DIR))
    args = parser.parse_args()
    print(f"Exported sentiment bundle: {export_model_bundle(args.source_dir, args.target_dir)}")


if __name__ == "__main__":
    main()
