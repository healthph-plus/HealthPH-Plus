"""Chronological and hand-calculated checks for post-count baselines."""

import numpy as np
import pandas as pd
import pytest

from healthph_plus.forecasting import BASELINES, baseline_forecast, evaluate_baselines


def daily_frame(values, *, start="2025-09-01", label="AURI"):
    values = np.asarray(values)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values), freq="D"),
        "disease_label": label,
        "post_count": values,
        "eligible_post_count": np.maximum(values, 1000),
    })


def four_diseases():
    return pd.concat([
        daily_frame(np.zeros(91)),
        daily_frame(np.full(91, 3), label="PN"),
        daily_frame(np.arange(91) % 7, label="TB"),
        daily_frame(np.arange(91), label="COVID"),
    ], ignore_index=True)


@pytest.mark.parametrize("model,expected", [
    ("zero", [0] * 10),
    ("last_value", [9] * 10),
    ("seasonal_naive_7", [3, 4, 5, 6, 7, 8, 9, 3, 4, 5]),
    ("mean_7", [6] * 10),
])
def test_baselines_against_hand_calculated_history(model, expected):
    result = baseline_forecast(np.arange(1, 10), model, horizon=10)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype.kind == "f"


def test_all_zero_histories_and_minimum_history():
    for model in BASELINES:
        np.testing.assert_array_equal(baseline_forecast([0] * 7, model), np.zeros(7))
    assert baseline_forecast([2], "last_value", horizon=1).tolist() == [2]
    assert baseline_forecast([2], "zero", horizon=1).tolist() == [0]
    for model in ("seasonal_naive_7", "mean_7"):
        with pytest.raises(ValueError, match="at least 7"):
            baseline_forecast([1] * 6, model)


@pytest.mark.parametrize("history", [[], [[1, 2]], [1, np.nan], [np.inf], [-1], ["bad"]])
def test_invalid_baseline_history(history):
    with pytest.raises(ValueError, match="history"):
        baseline_forecast(history, "last_value")


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_baseline_horizon(horizon):
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        baseline_forecast([0] * 7, "zero", horizon=horizon)


def test_unknown_baseline():
    with pytest.raises(ValueError, match="model must be one of"):
        baseline_forecast([0] * 7, "unknown")


def test_known_weekly_series_selects_seasonal_model_and_scores_exactly():
    result = evaluate_baselines(
        daily_frame(np.tile(np.arange(1, 8), 3)),
        initial_train_days=7, validation_folds=1, test_folds=1,
    )
    selected = result["selected_models"].iloc[0]
    assert selected["model"] == "seasonal_naive_7"
    assert selected["validation_mae"] == selected["validation_rmse"] == 0
    last_value = result["metrics"].query("split == 'validation' and model == 'last_value'").iloc[0]
    assert last_value["mae"] == 3
    assert last_value["rmse"] == pytest.approx(np.sqrt(13))
    assert last_value["n"] == 7
    assert result["metrics"].query("is_selected")["mae"].tolist() == [0, 0]
    np.testing.assert_array_equal(result["forecast"]["prediction"], np.arange(1, 8))
    assert result["forecast"]["actual"].isna().all()
    assert set(result["metrics"]["split"]) == {"validation", "test"}


def test_default_split_dates_horizons_counts_and_fixed_tie_order():
    result = evaluate_baselines(four_diseases().sample(frac=1, random_state=7))
    selected = result["selected_models"].set_index("disease_label")["model"].to_dict()
    assert selected == {"AURI": "zero", "PN": "last_value", "TB": "seasonal_naive_7", "COVID": "last_value"}
    folds = result["folds"]
    assert folds["split"].tolist() == ["validation"] * 3 + ["test"] * 2 + ["demonstration"]
    assert folds["fold"].tolist() == [1, 2, 3, 1, 2, 1]
    assert folds["train_days"].tolist() == [56, 63, 70, 77, 84, 91]
    assert folds["train_start"].eq(pd.Timestamp("2025-09-01")).all()
    assert folds["train_end"].tolist() == list(pd.to_datetime([
        "2025-10-26", "2025-11-02", "2025-11-09", "2025-11-16", "2025-11-23", "2025-11-30",
    ]))
    assert folds["target_start"].eq(folds["origin_date"] + pd.Timedelta(days=1)).all()
    assert folds["target_end"].eq(folds["origin_date"] + pd.Timedelta(days=7)).all()
    predictions = result["predictions"]
    assert len(predictions) == 5 * 4 * 4 * 7
    assert len(result["metrics"]) == 2 * 4 * 4
    assert len(result["horizon_metrics"]) == 2 * 4 * 4 * 7
    assert set(result["metrics"].query("split == 'validation'")["n"]) == {21}
    assert set(result["metrics"].query("split == 'test'")["n"]) == {14}
    for _, rows in predictions.groupby(["split", "fold", "disease_label", "model"]):
        assert rows["horizon"].tolist() == list(range(1, 8))
        assert rows["target_date"].is_monotonic_increasing
        assert (rows["target_date"] > rows["origin_date"]).all()
    for split in ["validation", "test"]:
        rows = predictions.query("split == @split")
        assert rows.query("is_selected").groupby("disease_label")["model"].nunique().eq(1).all()
    forecast = result["forecast"]
    assert len(forecast) == 4 * 7
    assert forecast["origin_date"].eq(pd.Timestamp("2025-11-30")).all()
    assert forecast["target_date"].min() == pd.Timestamp("2025-12-01")
    assert forecast["target_date"].max() == pd.Timestamp("2025-12-07")
    assert forecast["split"].eq("demonstration").all()
    assert forecast["is_selected"].all()


def test_validation_rmse_breaks_an_mae_tie_before_model_order():
    values = [2] * 7 + [0, 0, 0, 1, 2, 2, 5] + [0] * 7
    result = evaluate_baselines(daily_frame(values), initial_train_days=7, validation_folds=1, test_folds=1)
    validation = result["metrics"].query("split == 'validation'").set_index("model")
    assert validation.loc["zero", "mae"] == validation.loc["last_value", "mae"] == 10 / 7
    assert validation.loc["last_value", "rmse"] < validation.loc["zero", "rmse"]
    assert result["selected_models"].iloc[0]["model"] == "last_value"


def test_future_actuals_cannot_change_predictions_from_an_earlier_origin():
    original = four_diseases()
    changed = original.copy()
    changed.loc[changed["date"] >= pd.Timestamp("2025-11-24"), "post_count"] = 999
    before = evaluate_baselines(original)
    after = evaluate_baselines(changed)
    pd.testing.assert_frame_equal(before["predictions"].drop(columns="actual"), after["predictions"].drop(columns="actual"))
    assert not before["forecast"]["prediction"].equals(after["forecast"]["prediction"])


def test_test_data_cannot_change_selection_or_validation_predictions():
    original = four_diseases()
    changed = original.copy()
    changed.loc[changed["date"] >= pd.Timestamp("2025-11-17"), "post_count"] = 999
    before = evaluate_baselines(original)
    after = evaluate_baselines(changed)
    pd.testing.assert_frame_equal(before["selected_models"], after["selected_models"])
    pd.testing.assert_frame_equal(
        before["predictions"].query("split == 'validation'").reset_index(drop=True),
        after["predictions"].query("split == 'validation'").reset_index(drop=True),
    )
    # The later test origin may use actuals that were observed during the first test week.
    old_last = before["predictions"].query("split == 'test' and fold == 2 and model == 'last_value'")
    new_last = after["predictions"].query("split == 'test' and fold == 2 and model == 'last_value'")
    assert not old_last["prediction"].equals(new_last["prediction"])


def test_evaluation_does_not_modify_input():
    frame = four_diseases()
    original = frame.copy(deep=True)
    evaluate_baselines(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column,value", [
    ("date", None), ("date", "bad date"), ("disease_label", None), ("disease_label", " "),
    ("post_count", np.nan), ("post_count", np.inf), ("post_count", -1), ("post_count", 0.5),
    ("eligible_post_count", np.nan), ("eligible_post_count", -1),
])
def test_missing_or_invalid_observations_rejected(column, value):
    frame = daily_frame(np.zeros(91))
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate_baselines(frame)


def test_duplicate_dates_and_missing_dates_rejected():
    frame = daily_frame(np.zeros(91))
    with pytest.raises(ValueError, match="duplicate date/disease_label"):
        evaluate_baselines(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="expected exactly 91 daily observations.*got 90"):
        evaluate_baselines(frame.drop(index=10))
    frame.loc[10, "date"] = pd.Timestamp("2025-12-01")
    with pytest.raises(ValueError, match="contiguous daily"):
        evaluate_baselines(frame)


def test_extra_later_history_cannot_be_silently_truncated():
    with pytest.raises(ValueError, match="expected exactly 91 daily observations.*got 98"):
        evaluate_baselines(daily_frame(np.zeros(98)))


def test_disease_histories_require_identical_calendar_dates():
    frame = pd.concat([
        daily_frame(np.zeros(91)), daily_frame(np.zeros(91), start="2025-09-02", label="TB"),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="same calendar dates"):
        evaluate_baselines(frame)


@pytest.mark.parametrize("dates", [
    pd.date_range("2025-09-01", periods=91, tz="Asia/Manila"),
    pd.date_range("2025-09-01 12:00", periods=91),
])
def test_dates_must_be_local_midnight_without_timezone(dates):
    frame = daily_frame(np.zeros(91))
    frame["date"] = dates
    with pytest.raises(ValueError, match="timezone-naive daily dates at midnight"):
        evaluate_baselines(frame)


def test_counts_cannot_exceed_eligible_posts():
    frame = daily_frame(np.zeros(91))
    frame.loc[0, "post_count"] = 1001
    with pytest.raises(ValueError, match="post_count cannot exceed eligible_post_count"):
        evaluate_baselines(frame)


def test_zero_eligible_posts_cannot_be_treated_as_an_observed_zero():
    frame = daily_frame(np.zeros(91))
    frame.loc[0, "eligible_post_count"] = 0
    with pytest.raises(ValueError, match="zero eligible posts indicate unavailable observations"):
        evaluate_baselines(frame)


def test_eligible_denominator_must_agree_across_diseases_for_each_date():
    frame = four_diseases()
    frame.loc[91, "eligible_post_count"] = 999
    with pytest.raises(ValueError, match="identical across disease labels"):
        evaluate_baselines(frame)


@pytest.mark.parametrize("kwargs,message", [
    ({"initial_train_days": 6}, "at least 7"),
    ({"initial_train_days": 0}, "initial_train_days must be a positive integer"),
    ({"validation_folds": 0}, "validation_folds must be a positive integer"),
    ({"test_folds": 0}, "test_folds must be a positive integer"),
    ({"horizon": 0}, "horizon must be a positive integer"),
    ({"horizon": True}, "horizon must be a positive integer"),
])
def test_invalid_evaluation_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate_baselines(daily_frame(np.zeros(91)), **kwargs)


def test_empty_frame_and_missing_columns_rejected():
    with pytest.raises(ValueError, match="required columns"):
        evaluate_baselines(pd.DataFrame())
    with pytest.raises(ValueError, match="contain daily observations"):
        evaluate_baselines(daily_frame([]))
    with pytest.raises(ValueError, match="required columns.*eligible_post_count"):
        evaluate_baselines(daily_frame(np.zeros(91)).drop(columns="eligible_post_count"))
