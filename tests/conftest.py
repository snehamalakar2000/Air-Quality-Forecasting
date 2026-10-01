"""Small, known-value sources with independent provenance records."""

import pandas as pd
import pytest

from air_quality.data import sha256
from air_quality.pipeline import synthetic_fixture


@pytest.fixture
def raw_hours():
    time = pd.date_range("2012-12-31", periods=60, freq="h")
    return pd.DataFrame({"No": range(1, 61), "year": time.year, "month": time.month,
                         "day": time.day, "hour": time.hour,
                         "pm2.5": [float(x) for x in range(60)],
                         "DEWP": -3, "TEMP": -2, "PRES": 1010,
                         "cbwd": "NW", "Iws": 0, "Is": 0, "Ir": 0})


@pytest.fixture
def save_source(tmp_path):
    def save(raw):
        path = tmp_path / "fixture.csv"
        raw.to_csv(path, index=False, na_rep="NA")
        time = pd.to_datetime(raw[["year", "month", "day", "hour"]], errors="coerce")
        source = {"sha256": sha256(path), "size_bytes": path.stat().st_size,
                  "rows": len(raw), "start": str(time.min()), "end": str(time.max())}
        return path, source
    return save


@pytest.fixture
def small_config(tmp_path):
    return synthetic_fixture(tmp_path / "synthetic")
