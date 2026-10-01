"""Causal inputs and future labels are deliberately separate functions."""

import numpy as np
import pandas as pd

from .data import WEATHER

HOUR = pd.Timedelta(hours=1)


def check_hourly(frame):
    index = frame.index
    if index.tz is None or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Expected a sorted, unique timezone-aware hourly grid")
    if len(index) > 1 and not ((index[1:] - index[:-1]) == HOUR).all():
        raise ValueError("Reindex to the complete hourly grid before shifts")


def build_features(frame, settings):
    """Each row uses measurements through its origin, never through its target."""
    check_hourly(frame)
    pm = frame.pm25
    inputs = pd.DataFrame(index=frame.index)
    inputs["pm25_current"] = pm
    for lag in settings["lags"]:
        inputs[f"pm25_lag_{lag}"] = pm.shift(lag)
    for spec in settings["rolling"]:
        hours, minimum = spec["hours"], spec["min_observed"]
        inputs[f"pm25_mean_{hours}"] = pm.rolling(hours, min_periods=minimum).mean()
        inputs[f"history_count_{hours}"] = pm.rolling(hours, min_periods=0).count()
    inputs["pm25_change_1"] = pm - pm.shift(1)
    inputs[WEATHER] = frame[WEATHER]
    target_time = frame.index + HOUR
    inputs["target_hour_sin"] = np.sin(2 * np.pi * target_time.hour / 24)
    inputs["target_hour_cos"] = np.cos(2 * np.pi * target_time.hour / 24)
    inputs["target_month_sin"] = np.sin(2 * np.pi * (target_time.month - 1) / 12)
    inputs["target_month_cos"] = np.cos(2 * np.pi * (target_time.month - 1) / 12)
    inputs["target_day_of_week"] = target_time.dayofweek
    # Fixed flags exist even when a column is complete in training.
    for column in list(inputs.columns):
        inputs[f"{column}_missing"] = inputs[column].isna().astype(float)
    inputs["cbwd"] = frame.cbwd.fillna("missing")
    return inputs


def build_labels(frame):
    check_hourly(frame)
    audit = pd.DataFrame(index=frame.index)
    audit["origin_time"] = frame.index
    audit["target_time"] = frame.index + HOUR
    audit["actual"] = frame.pm25.shift(-1)
    audit["current_pm25"] = frame.pm25
    audit["current_row_absent"] = ~frame.row_present
    audit["current_pm25_missing"] = frame.pm25.isna()
    # Reindex by exact target timestamp. The final origin has no source target.
    present = frame.row_present.reindex(frame.index + HOUR, fill_value=False)
    audit["target_hour_absent"] = ~present.to_numpy(dtype=bool)
    audit["target_pm25_missing"] = audit.actual.isna()
    audit["eligible"] = ~(audit.current_pm25_missing | audit.target_pm25_missing |
                          audit.current_row_absent | audit.target_hour_absent)
    return audit


def split_data(inputs, audit, splits, timezone):
    partitions = {}
    coverage = {}
    reasons = ["current_row_absent", "current_pm25_missing", "target_hour_absent",
               "target_pm25_missing"]
    for name, (start, end) in splits.items():
        start, end = pd.Timestamp(start, tz=timezone), pd.Timestamp(end, tz=timezone)
        mask = (audit.target_time >= start) & (audit.target_time < end) & audit.eligible
        partitions[name] = (inputs.loc[mask].copy(), audit.loc[mask].copy())
        # Include the first source hour, which has no preceding origin, in coverage.
        targets = pd.date_range(start, end, freq="h", inclusive="left")
        population = audit.set_index("target_time").reindex(targets)
        no_origin = population.origin_time.isna()
        valid = population.eligible.eq(True)
        coverage[name] = {
            "possible_target_hours": len(targets), "eligible_rows": int(valid.sum()),
            "forecast_coverage": float((~(population.current_pm25_missing.ne(False) |
                                          no_origin)).mean()),
            "scoring_coverage": float(valid.mean()),
            "total_unique_excluded_origins": int((~valid).sum()),
            "exclusions_by_reason": {reason: int(population[reason].eq(True).sum())
                                     for reason in reasons},
            "no_origin_in_source": int(no_origin.sum()),
        }
    return partitions, coverage
