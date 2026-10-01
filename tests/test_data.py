import numpy as np
import pandas as pd
import pytest

from air_quality.data import load_data


def test_unsorted_gap_report_and_high_values(raw_hours, save_source):
    raw_hours.loc[5, "pm2.5"] = 1800
    raw_hours.loc[7:9, "pm2.5"] = np.nan
    raw_hours.loc[12, "cbwd"] = "unexpected"
    raw = raw_hours.drop(index=[10, 11]).iloc[::-1]
    frame, report = load_data(*save_source(raw))
    assert frame.index.is_monotonic_increasing
    assert report["reordering_required"] is True
    assert report["absent_hours"] == 2
    assert report["longest_missing_run_hours"]["pm25"] == 5
    assert report["unusual_wind_categories"] == {"unexpected": 1}
    assert frame.pm25.max() == 1800
    assert frame.iloc[10].row_present == False
    sorted_frame, _ = load_data(*save_source(raw.sort_values("No")))
    pd.testing.assert_frame_equal(frame, sorted_frame)


@pytest.mark.parametrize("column,value,message", [
    ("TEMP", "oops", "Malformed numeric"), ("TEMP", np.inf, "Infinite"),
    ("pm2.5", -1, "Negative"), ("Iws", -1, "Negative"),
    ("Is", -1, "Negative"), ("Ir", -1, "Negative"), ("PRES", 0, "positive"),
    ("hour", 24, "Hour"), ("month", 2.5, "integers"),
    ("day", 32, "Invalid calendar"), ("year", np.nan, "integers")])
def test_invalid_observations(raw_hours, save_source, column, value, message):
    # Object conversion permits intentionally invalid text without a pandas warning.
    raw_hours[column] = raw_hours[column].astype(object)
    raw_hours.loc[2, column] = value
    with pytest.raises(ValueError, match=message):
        load_data(*save_source(raw_hours))


def test_duplicate_schema_checksum_and_provenance(raw_hours, save_source):
    duplicate = pd.concat([raw_hours, raw_hours.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        load_data(*save_source(duplicate))
    path, source = save_source(raw_hours.drop(columns="TEMP"))
    with pytest.raises(ValueError, match="required columns"):
        load_data(path, source)
    path, source = save_source(raw_hours)
    with pytest.raises(ValueError, match="checksum"):
        load_data(path, {**source, "sha256": "wrong"})
    with pytest.raises(ValueError, match="byte size"):
        load_data(path, {**source, "size_bytes": 1})
    with pytest.raises(ValueError, match="row count"):
        load_data(path, {**source, "rows": 1})
    with pytest.raises(ValueError, match="date range"):
        load_data(path, {**source, "start": "2010-01-01"})
