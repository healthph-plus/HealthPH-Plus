"""Verify forecast source tables and raster/vector report artifacts agree."""

import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from healthph_plus.forecast_data import LABELS
from healthph_plus.forecast_reporting import export_forecast_report
from healthph_plus.forecasting import evaluate_baselines


def test_complete_forecasting_report_preserves_predictions_and_unknown_actuals(tmp_path):
    dates = pd.date_range("2025-09-01", periods=91)
    daily = pd.concat([
        pd.DataFrame({"date": dates, "disease_label": label,
                      "post_count": (np.arange(91) + index) % 7,
                      "eligible_post_count": 20})
        for index, label in enumerate(LABELS)
    ], ignore_index=True)
    sources = pd.concat([
        pd.DataFrame({"date": dates, "source": source, "post_count": 10})
        for source in ("reddit", "twitter")
    ], ignore_index=True)
    # Match the nullable integer counts returned by the real aggregation path.
    sources["post_count"] = sources["post_count"].astype("Int64")
    evaluation = evaluate_baselines(daily)
    artifacts = export_forecast_report(daily, sources, evaluation, tmp_path, show=False)
    assert len(artifacts) == 24  # Eight CSV/JSON pairs, four PNG/SVG pairs.
    for name in ("daily_counts_and_sources", "baseline_comparison", "horizon_errors", "retrospective_forecasts"):
        for suffix in ("png", "svg"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 1000
    saved = pd.read_csv(tmp_path / "predictions.csv")
    np.testing.assert_allclose(saved["prediction"], evaluation["predictions"]["prediction"])
    np.testing.assert_allclose(saved["actual"], evaluation["predictions"]["actual"])
    assert len(saved) == 560
    saved_metrics = pd.read_csv(tmp_path / "metrics.csv")
    np.testing.assert_allclose(saved_metrics["mae"], evaluation["metrics"]["mae"])
    np.testing.assert_allclose(saved_metrics["rmse"], evaluation["metrics"]["rmse"])
    future = json.loads((tmp_path / "forecast.json").read_text())
    assert len(future) == 28
    assert all(row["actual"] is None for row in future)
    assert all(row["split"] == "demonstration" for row in future)
    assert len(json.loads((tmp_path / "daily_series.json").read_text())) == 364
