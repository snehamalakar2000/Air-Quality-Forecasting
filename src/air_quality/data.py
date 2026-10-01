"""Validate the immutable local source before constructing any features."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["No", "year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP",
            "PRES", "cbwd", "Iws", "Is", "Ir"]
CALENDAR = ["year", "month", "day", "hour"]
WEATHER = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]


def sha256(path):
    """Hash file bytes, rather than a potentially reformatted DataFrame."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def longest_run(mask):
    longest = current = 0
    for missing in mask:
        current = current + 1 if missing else 0
        longest = max(longest, current)
    return longest


def load_data(path, source, timezone="Asia/Shanghai"):
    path = Path(path)
    actual_hash = sha256(path)
    if actual_hash != source["sha256"]:
        raise ValueError("Dataset checksum mismatch; restore the approved source CSV")
    if path.stat().st_size != source["size_bytes"]:
        raise ValueError("Dataset byte size differs from provenance")
    # Preserve text first: malformed numeric tokens must fail, never become NA.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["NA", ""])
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Source CSV has no rows")
    for column in set(REQUIRED) - {"cbwd"}:
        try:
            raw[column] = pd.to_numeric(raw[column], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Malformed numeric value in {column}") from error
        if np.isinf(raw[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite numeric value in {column}")
    for column in CALENDAR:
        if raw[column].isna().any() or (raw[column] % 1 != 0).any():
            raise ValueError(f"Calendar field {column} must contain integers")
    if not raw["hour"].between(0, 23).all():
        raise ValueError("Hour must be between 0 and 23")
    try:
        time = pd.DatetimeIndex(pd.to_datetime(raw[CALENDAR], errors="raise"))
        time = time.tz_localize(timezone)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid calendar date") from error
    if time.has_duplicates:
        raise ValueError("Duplicate timestamps")
    for column in ["pm2.5", "Iws", "Is", "Ir"]:
        if (raw[column].dropna() < 0).any():
            raise ValueError(f"Negative observed {column}")
    if (raw["PRES"].dropna() <= 0).any():
        raise ValueError("Pressure must be positive")
    was_sorted = time.is_monotonic_increasing
    raw.index = time
    raw.index.name = "origin_time"
    raw = raw.sort_index().rename(columns={"pm2.5": "pm25"})
    if len(raw) != source["rows"]:
        raise ValueError("Source row count differs from provenance")
    for actual, expected in [(raw.index.min(), source["start"]),
                             (raw.index.max(), source["end"])]:
        if actual != pd.Timestamp(expected, tz=timezone):
            raise ValueError("Source date range differs from provenance")
    raw["row_present"] = True
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="h", name="origin_time")
    frame = raw.reindex(grid)
    frame["row_present"] = frame["row_present"].eq(True)
    unexpected = raw.loc[~raw["cbwd"].isin(["NE", "NW", "SE", "cv"]) &
                         raw["cbwd"].notna(), "cbwd"].value_counts()
    report = {
        "sha256": actual_hash, "size_bytes": path.stat().st_size,
        "raw_rows": len(raw), "hourly_grid_rows": len(frame),
        "start": str(grid.min()), "end": str(grid.max()),
        "reordering_required": not was_sorted,
        "absent_hours": int((~frame.row_present).sum()),
        "raw_missing_per_column": {c: int(raw[c].isna().sum()) for c in raw.columns},
        "grid_missing_per_column": {c: int(frame[c].isna().sum()) for c in raw.columns},
        "longest_missing_run_hours": {c: longest_run(frame[c].isna())
                                      for c in raw.columns if c != "row_present"},
        "longest_absent_run_hours": longest_run(~frame.row_present),
        "unusual_wind_categories": {str(k): int(v) for k, v in unexpected.items()},
    }
    return frame, report
