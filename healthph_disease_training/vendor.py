"""Refresh the shared training runtime in both standalone service folders."""

from pathlib import Path
import shutil

RUNTIME_FILES = ("__init__.py", "data.py", "runner.py")


def vendor_runtime(service):
    service = Path(service).resolve()
    if not any((service / module).is_file() for module in ("healthph_plus/train.py", "healthph_xlmr/train.py")):
        raise ValueError(f"Not a disease training service: {service}")
    target = service / "healthph_disease_training"
    target.mkdir(exist_ok=True)
    for name in RUNTIME_FILES:
        shutil.copy2(Path(__file__).parent / name, target / name)
    (target / "GENERATED.txt").write_text(
        "Generated from the repository healthph_disease_training package. Do not edit this copy.\n"
        "Refresh both services from the repository root: python -m healthph_disease_training.vendor\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name in ("disease-detection-service", "xlmr-disease-detection-service"):
        vendor_runtime(root / name)
