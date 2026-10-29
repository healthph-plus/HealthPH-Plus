"""Tables and figures for retrospective daily disease-post forecasts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from healthph_plus.forecast_data import LABELS
from healthph_plus.forecasting import BASELINES
from healthph_plus.model_reporting import save_figure, save_table


def _shade_periods(ax, folds):
    for split, color in (("validation", "#E69F00"), ("test", "#009E73")):
        period = folds.loc[folds["split"].eq(split)]
        if not period.empty:
            start = pd.to_datetime(period["target_start"]).min()
            end = pd.to_datetime(period["target_end"]).max() + pd.Timedelta(days=1)
            ax.axvspan(start, end, color=color, alpha=0.10, label=split.title())


def export_forecast_report(daily_series, daily_sources, evaluation, report_dir, *, show=True):
    """Export evaluation source tables and four figures; return their paths.

    Raw post text is never included. Missing actuals in the demonstration are
    preserved as CSV blanks / JSON nulls by the shared reporting utilities.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = {"daily_series": daily_series, "daily_sources": daily_sources, **evaluation}
    for name, frame in tables.items():
        save_table(frame, report_dir / name)

    folds = evaluation["folds"]
    selected = evaluation["selected_models"].set_index("disease_label")["model"].to_dict()
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True, layout="constrained")
    sources = daily_sources.pivot(index="date", columns="source", values="post_count").sort_index()
    if not sources.empty:
        sources.index = pd.to_datetime(sources.index)
        # Aggregation returns pandas nullable Int64; matplotlib needs numeric
        # NumPy arrays rather than the extension array's default object dtype.
        axes[0].stackplot(sources.index, sources.to_numpy(dtype=float, na_value=np.nan).T,
                          labels=sources.columns, alpha=0.8)
    axes[0].set(title="Eligible posts by source", ylabel="Posts")
    for ax, label in zip(axes[1:], LABELS):
        frame = daily_series.loc[daily_series["disease_label"].eq(label)]
        ax.plot(pd.to_datetime(frame["date"]), frame["post_count"].astype(float), color="#0072B2")
        ax.set(ylabel="Posts", title=label)
    for ax in axes:
        _shade_periods(ax, folds)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Posting date (Asia/Manila)")
    fig.suptitle("Sampled corpus: source composition and frozen XLM-RoBERTa disease signals")
    save_figure(fig, report_dir / "daily_counts_and_sources", show=show)

    metrics = evaluation["metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    positions = np.arange(len(BASELINES))
    for ax, label in zip(axes.flat, LABELS):
        for offset, split, color in ((-0.18, "validation", "#E69F00"), (0.18, "test", "#009E73")):
            values = metrics.loc[metrics["disease_label"].eq(label) & metrics["split"].eq(split)]
            values = values.set_index("model").reindex(BASELINES)
            ax.bar(positions + offset, values["mae"], width=0.36, color=color, label=split.title())
        names = [name + (" *" if selected.get(label) == name else "") for name in BASELINES]
        ax.set_xticks(positions, names, rotation=18, ha="right")
        ax.set(title=label, ylabel="Mean absolute error (posts)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Baseline comparison (* selected using validation MAE, then RMSE)")
    save_figure(fig, report_dir / "baseline_comparison", show=show)

    horizon_scores = evaluation["horizon_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for label in LABELS:
        values = horizon_scores.loc[
            horizon_scores["split"].eq("test")
            & horizon_scores["disease_label"].eq(label)
            & horizon_scores["model"].eq(selected[label])
        ].sort_values("horizon")
        for ax, metric in zip(axes, ("mae", "rmse")):
            ax.plot(values["horizon"], values[metric], marker="o", label=label)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        ax.set(title=metric, xlabel="Days ahead", ylabel="Error (posts)")
        ax.set_xticks(sorted(horizon_scores["horizon"].unique()))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Selected models: test error by forecast horizon (two weekly origins)")
    save_figure(fig, report_dir / "horizon_errors", show=show)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for ax, label in zip(axes.flat, LABELS):
        actuals = daily_series.loc[daily_series["disease_label"].eq(label)]
        ax.plot(pd.to_datetime(actuals["date"]), actuals["post_count"].astype(float),
                color="#555555", linewidth=1.2, label="Observed sampled posts")
        for split, color in (("validation", "#E69F00"), ("test", "#009E73")):
            values = evaluation["predictions"]
            values = values.loc[values["split"].eq(split) & values["disease_label"].eq(label)
                                & values["model"].eq(selected[label])].sort_values("target_date")
            ax.plot(pd.to_datetime(values["target_date"]), values["prediction"], color=color,
                    linewidth=1.6, label=f"{split.title()} forecast")
        future = evaluation["forecast"]
        future = future.loc[future["disease_label"].eq(label)].sort_values("target_date")
        ax.plot(pd.to_datetime(future["target_date"]), future["prediction"], "o--",
                color="#CC79A7", markersize=3, label="Retrospective demonstration")
        _shade_periods(ax, folds)
        ax.set(title=f"{label}: {selected[label]}", ylabel="Expected sampled posts",
               xlabel="Posting date (Asia/Manila)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        # The shaded periods repeat the line labels' information.
        handles, names = ax.get_legend_handles_labels()
        ax.legend(handles[:4], names[:4], fontsize=7, loc="upper left")
    fig.suptitle("Historical forecasting experiment; demonstration follows the selected window")
    save_figure(fig, report_dir / "retrospective_forecasts", show=show)
    return sorted(path for path in report_dir.iterdir() if path.is_file())
