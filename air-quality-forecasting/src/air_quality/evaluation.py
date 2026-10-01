"""Shared scoring populations, honest error slices, and three fixed figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import persistence, predict

METHODS = ["persistence", "ridge", "forest"]
LABELS = {"persistence": "Persistence", "ridge": "Ridge", "forest": "Random Forest"}


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have matching shapes")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Metrics require finite observed targets and predictions")
    if not len(actual):
        return {"count": 0, "mae": None, "rmse": None, "mean_signed_error": None,
                "fraction_underpredicted": None}
    errors = predicted - actual
    return {"count": len(actual), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean_signed_error": float(errors.mean()),
            "fraction_underpredicted": float((errors < 0).mean())}


def prediction_table(inputs, audit, models):
    if not inputs.index.equals(audit.index):
        raise ValueError("Common scoring population has misaligned inputs and labels")
    if not audit.eligible.all() or not ((audit.target_time - audit.origin_time) ==
                                      pd.Timedelta(hours=1)).all():
        raise ValueError("Scoring requires eligible exact-hour targets")
    table = audit[["origin_time", "target_time", "actual", "current_pm25"]].copy()
    for column in inputs:
        if column.startswith("history_count_"):
            table[column] = inputs[column]
    table["persistence"] = persistence(inputs)
    for kind in ["ridge", "forest"]:
        table[kind] = predict(models[kind], inputs)
    for kind in METHODS:
        table[f"{kind}_signed_error"] = table[kind] - table.actual
    return table.reset_index(drop=True)


def error_slices(table, threshold):
    masks = {"overall": np.ones(len(table), dtype=bool),
             "high_pollution": table.actual >= threshold,
             "remaining": table.actual < threshold}
    return {name: {kind: metrics(table.loc[mask, "actual"], table.loc[mask, kind])
                   for kind in METHODS} for name, mask in masks.items()}


def fixed_period_grid(table, period, timezone):
    start, end = [pd.Timestamp(value, tz=timezone) for value in period]
    grid = pd.date_range(start, end, freq="h", inclusive="left", name="target_time")
    if table.target_time.duplicated().any():
        raise ValueError("Duplicate scored target timestamps")
    # NaN rows break lines at missing hours rather than bridging sensor outages.
    return table.set_index("target_time").reindex(grid)


def plot_reports(frame, table, slices, threshold, selected, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140,
                         "timezone": config["timezone"]})
    start, end = [pd.Timestamp(t, tz=config["timezone"]) for t in config["splits"]["train"]]
    training = frame.loc[(frame.index >= start) & (frame.index < end)]
    monthly = training.pm25.resample("MS").mean()
    absent = (~training.row_present).astype(int).resample("MS").sum()
    sensor_missing = (training.row_present & training.pm25.isna()).astype(int).resample("MS").sum()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].plot(monthly.index, monthly, color="#227c9d", marker="o", markersize=3)
    axes[0].set(ylabel="Observed mean PM2.5 (µg/m³)", title="Training pollution and missingness · before imputation")
    axes[1].bar(absent.index, absent, width=20, label="Absent timestamp rows", color="#db7c26")
    axes[1].bar(sensor_missing.index, sensor_missing, bottom=absent, width=20,
                label="Present rows, missing PM2.5", color="#64748b")
    axes[1].set(ylabel="Missing hours", xlabel="Month · Beijing local time")
    axes[1].legend()
    fig.savefig(directory / "training_overview.png")
    plt.close(fig)

    window = fixed_period_grid(table, config["fixed_test_period"], config["timezone"])
    count = int(window.actual.notna().sum())
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for column, label, color in [("actual", "Actual", "#111827"),
                                  ("persistence", "Persistence", "#db7c26"),
                                  (selected, f"{LABELS[selected]} · validation-selected", "#227c9d")]:
        ax.plot(window.index, window[column], label=label, color=color, linewidth=1.3)
    ax.set(title=f"Fixed test period · {count}/{len(window)} scored hours",
           xlabel="Target time · Beijing local time", ylabel="PM2.5 (µg/m³)")
    if count == 0:
        ax.text(0.5, 0.5, "No eligible scored hours in the configured window",
                transform=ax.transAxes, ha="center")
    ax.legend()
    fig.savefig(directory / "test_predictions_fixed_period.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    colors = ["#64748b", "#db7c26", "#227c9d"]
    groups = list(slices)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        x = np.arange(len(METHODS))
        for offset, (group, color) in enumerate(zip(groups, colors)):
            count = slices[group]["persistence"]["count"]
            values = [slices[group][kind][metric] for kind in METHODS]
            if count:
                ax.bar(x + (offset - 1) * 0.24, values, width=0.24, color=color,
                       label=f"{group.replace('_', ' ')} · n={count:,}")
            else:
                ax.plot([], [], color=color, label=f"{group} · n=0 (bars omitted)")
        ax.set_xticks(x, [LABELS[k] + ("\n(validation-selected)" if k == selected else "")
                         for k in METHODS])
        ax.set(ylabel=f"{metric.upper()} (µg/m³)", title=metric.upper())
        maximum = max(slices[group][kind][metric] or 0 for group in groups for kind in METHODS)
        ax.set_ylim(0, max(1, maximum * 1.25))
        ax.legend(fontsize=8)
    fig.suptitle(f"Final test errors · high pollution ≥ {threshold:.2f} µg/m³\n"
                 "Threshold = training target 90th percentile; selection frozen on validation")
    fig.savefig(directory / "model_error_comparison.png")
    plt.close(fig)
