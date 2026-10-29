"""Small, chronological baselines for forecasting model-labeled post counts.

This module uses only NumPy and pandas. It does not load a disease classifier or
interpret the resulting post counts as disease cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASELINES = ("zero", "last_value", "seasonal_naive_7", "mean_7")
_PREDICTION_COLUMNS = [
    "origin_date", "target_date", "disease_label", "model", "horizon",
    "prediction", "actual", "split", "fold", "is_selected",
]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def baseline_forecast(history, model, *, horizon=7) -> np.ndarray:
    """Forecast from a nonempty, finite, nonnegative one-dimensional history.

    ``seasonal_naive_7`` repeats the last observed week, including when the
    requested horizon exceeds seven days. ``mean_7`` repeats that week's mean.
    Those two models require at least seven observations; no method fills missing
    observations or updates itself with future actuals inside the horizon.
    """
    if model not in BASELINES:
        raise ValueError(f"model must be one of {BASELINES}")
    horizon = _positive_integer(horizon, "horizon")
    try:
        values = np.asarray(history, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("history must be a one-dimensional sequence of finite nonnegative numbers") from exc
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("history must be a nonempty one-dimensional sequence of finite nonnegative numbers")
    if model in ("seasonal_naive_7", "mean_7") and len(values) < 7:
        raise ValueError(f"{model} requires at least 7 history observations")
    if model == "zero":
        return np.zeros(horizon, dtype=float)
    if model == "last_value":
        return np.full(horizon, values[-1], dtype=float)
    if model == "seasonal_naive_7":
        return np.resize(values[-7:], horizon).astype(float)
    return np.full(horizon, values[-7:].mean(), dtype=float)


def _validate_daily_series(daily_series, expected_days):
    required = {"date", "disease_label", "post_count", "eligible_post_count"}
    if not isinstance(daily_series, pd.DataFrame):
        raise ValueError("daily_series must be a pandas DataFrame")
    missing = required - set(daily_series.columns)
    if missing:
        raise ValueError(f"daily_series is missing required columns: {sorted(missing)}")
    if daily_series.empty:
        raise ValueError("daily_series must contain daily observations")
    frame = daily_series.loc[:, ["date", "disease_label", "post_count", "eligible_post_count"]].copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", format="mixed")
        dates = pd.DatetimeIndex(frame["date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("date must contain valid timezone-naive daily dates") from exc
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("date must contain nonmissing timezone-naive daily dates at midnight")
    if not frame["disease_label"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("disease_label must contain nonempty strings")
    if frame.duplicated(["date", "disease_label"]).any():
        raise ValueError("duplicate date/disease_label observations are not allowed")
    for column in ("post_count", "eligible_post_count"):
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain finite nonnegative integer counts") from exc
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite nonnegative integer counts")
        frame[column] = values
    if (frame["eligible_post_count"] <= 0).any():
        raise ValueError("eligible_post_count must be positive; zero eligible posts indicate unavailable observations")
    if (frame["post_count"] > frame["eligible_post_count"]).any():
        raise ValueError("post_count cannot exceed eligible_post_count")
    if (frame.groupby("date")["eligible_post_count"].nunique() > 1).any():
        raise ValueError("eligible_post_count must be identical across disease labels for each date")

    histories, calendar = {}, None
    # Preserve the caller's disease order while sorting each history by date.
    for label, group in frame.groupby("disease_label", sort=False):
        group = group.sort_values("date")
        label_dates = pd.DatetimeIndex(group["date"])
        if len(group) != expected_days:
            raise ValueError(
                f"{label}: expected exactly {expected_days} daily observations for the configured "
                f"training/validation/test windows; got {len(group)}. "
                "Select the intended window explicitly or change the fold configuration."
            )
        contiguous = pd.date_range(label_dates[0], periods=expected_days, freq="D")
        if not label_dates.equals(contiguous):
            raise ValueError(f"{label}: history must be contiguous daily observations with no missing dates")
        if calendar is not None and not label_dates.equals(calendar):
            raise ValueError("all disease histories must cover the same calendar dates")
        calendar = label_dates
        histories[label] = group["post_count"].to_numpy(dtype=float)
    return calendar, histories


def _metric_table(predictions, *, by_horizon=False):
    keys = ["split", "disease_label", "model"]
    if by_horizon:
        keys.append("horizon")
    records = []
    for key, group in predictions.groupby(keys, sort=False):
        errors = group["prediction"].to_numpy() - group["actual"].to_numpy()
        records.append({
            **dict(zip(keys, key)),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
            "n": len(group),
            "is_selected": bool(group["is_selected"].iloc[0]),
        })
    return pd.DataFrame(records, columns=[*keys, "mae", "rmse", "n", "is_selected"])


def evaluate_baselines(
    daily_series, *, initial_train_days=56, validation_folds=3,
    test_folds=2, horizon=7,
) -> dict[str, pd.DataFrame]:
    """Run expanding chronological validation/test folds and a final demonstration.

    Input has one row per date/disease and columns ``date``, ``disease_label``,
    ``post_count``, and ``eligible_post_count``. Counts are observed nonnegative
    integers; eligible counts must be positive and identical across diseases on
    each date. Missing dates/counts and zero eligible counts are errors. Every disease must cover the same
    contiguous window of exactly ``initial_train_days + horizon *
    (validation_folds + test_folds)`` days. Extra history is never truncated.

    Select each disease's model by pooled validation MAE, then RMSE, then the
    order in ``BASELINES``. Freeze selection before evaluating any test folds.
    A test origin may use actuals from earlier test folds, as they are already
    observed at that origin. Each forecast within a fold uses its origin's
    history only. The returned ``forecast`` is an unscored demonstration after
    the final observed date; it is excluded from evaluation metrics.
    """
    initial_train_days = _positive_integer(initial_train_days, "initial_train_days")
    validation_folds = _positive_integer(validation_folds, "validation_folds")
    test_folds = _positive_integer(test_folds, "test_folds")
    horizon = _positive_integer(horizon, "horizon")
    if initial_train_days < 7:
        raise ValueError("initial_train_days must be at least 7 for weekly baselines")
    total_days = initial_train_days + horizon * (validation_folds + test_folds)
    dates, histories = _validate_daily_series(daily_series, total_days)
    prediction_rows, fold_rows = [], []

    def fold_metadata(split, fold, train_days, targets):
        return {
            "split": split, "fold": fold, "origin_date": dates[train_days - 1],
            "train_start": dates[0], "train_end": dates[train_days - 1],
            "target_start": targets[0], "target_end": targets[-1],
            "train_days": train_days, "target_days": horizon,
        }

    def append_fold(split, fold, train_days):
        targets = dates[train_days:train_days + horizon]
        fold_rows.append(fold_metadata(split, fold, train_days, targets))
        for label, values in histories.items():
            for model in BASELINES:
                predictions = baseline_forecast(values[:train_days], model, horizon=horizon)
                for step, (target_date, predicted, actual) in enumerate(
                    zip(targets, predictions, values[train_days:train_days + horizon]), start=1,
                ):
                    prediction_rows.append({
                        "origin_date": dates[train_days - 1], "target_date": target_date,
                        "disease_label": label, "model": model, "horizon": step,
                        "prediction": float(predicted), "actual": float(actual),
                        "split": split, "fold": fold, "is_selected": False,
                    })

    for fold in range(1, validation_folds + 1):
        append_fold("validation", fold, initial_train_days + (fold - 1) * horizon)

    validation = pd.DataFrame(prediction_rows, columns=_PREDICTION_COLUMNS)
    validation_metrics = _metric_table(validation)
    selected_rows, selected = [], {}
    for label in histories:
        candidates = validation_metrics.loc[validation_metrics["disease_label"] == label].copy()
        candidates["model_order"] = candidates["model"].map(BASELINES.index)
        best = candidates.sort_values(["mae", "rmse", "model_order"]).iloc[0]
        selected[label] = best["model"]
        selected_rows.append({
            "disease_label": label, "model": best["model"],
            "validation_mae": float(best["mae"]), "validation_rmse": float(best["rmse"]),
        })

    # Test actuals cannot affect model selection above.
    for fold in range(1, test_folds + 1):
        train_days = initial_train_days + (validation_folds + fold - 1) * horizon
        append_fold("test", fold, train_days)

    predictions = pd.DataFrame(prediction_rows, columns=_PREDICTION_COLUMNS)
    predictions["is_selected"] = predictions["model"].eq(predictions["disease_label"].map(selected))
    metrics = _metric_table(predictions)
    horizon_metrics = _metric_table(predictions, by_horizon=True)

    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    fold_rows.append(fold_metadata("demonstration", 1, total_days, future_dates))
    forecast_rows = []
    for label, values in histories.items():
        model = selected[label]
        for step, (target_date, predicted) in enumerate(
            zip(future_dates, baseline_forecast(values, model, horizon=horizon)), start=1,
        ):
            forecast_rows.append({
                "origin_date": dates[-1], "target_date": target_date,
                "disease_label": label, "model": model, "horizon": step,
                "prediction": float(predicted), "actual": np.nan,
                "split": "demonstration", "fold": 1, "is_selected": True,
            })
    return {
        "predictions": predictions,
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "selected_models": pd.DataFrame(selected_rows),
        "forecast": pd.DataFrame(forecast_rows, columns=_PREDICTION_COLUMNS),
        "folds": pd.DataFrame(fold_rows),
    }
