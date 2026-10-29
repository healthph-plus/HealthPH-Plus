"""Measure CPU disease inference with the default keyword extractor."""

import argparse
import importlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time


def benchmark_service(service_dir, bundle_dir=None, texts=None, repeats=5):
    from .export import _service_module

    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    service = Path(service_dir).resolve()
    bundle = Path(bundle_dir).resolve() if bundle_dir else service / "model"
    texts = texts if texts is not None else ["ubo at sipon", "walang ubo pero walang panlasa"]
    sys.path.insert(0, str(service))
    module = importlib.import_module(_service_module(service))
    loaded = time.perf_counter()
    predictor = module.DiseasePredictor(bundle)
    load_seconds = time.perf_counter() - loaded
    predictor.predict_many(texts)
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        predictor.predict_many(texts)
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    # macOS ru_maxrss is bytes; Linux uses KiB. This is whole-process peak RSS,
    # not a claimed model-only memory measurement.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
    return {
        "device": "cpu", "batch_size": len(texts), "repeats": repeats,
        "load_seconds": load_seconds, "batch_latency_p50_seconds": statistics.median(durations),
        "batch_latency_p95_seconds": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
        "whole_process_peak_rss_mib": peak_mib,
        "model_version": predictor.model_version,
        "symptom_model_version": predictor.symptom_model_version,
        "note": "Measure representative long posts and batch sizes when sizing client timeouts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-dir", required=True)
    parser.add_argument("--bundle-dir")
    parser.add_argument("--texts-json", help="UTF-8 JSON array of representative cleaned texts")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    texts = json.loads(Path(args.texts_json).read_text()) if args.texts_json else None
    report = benchmark_service(args.service_dir, args.bundle_dir, texts, args.repeats)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
