import numpy as np
import pandas as pd
import pytest

from air_quality.data import load_data
from air_quality.features import build_features, build_labels, split_data
from air_quality.models import make_model, forecast
from air_quality.pipeline import prepare

SETTINGS = {"lags": [1, 2, 3, 6, 12, 24],
            "rolling": [{"hours": 6, "min_observed": 3}, {"hours": 24, "min_observed": 12}]}


def test_manual_values_partial_history_and_gap(raw_hours, save_source):
    frame, _ = load_data(*save_source(raw_hours))
    x = build_features(frame, SETTINGS)
    audit = build_labels(frame)
    assert x.iloc[6].pm25_lag_1 == 5
    assert x.iloc[6].pm25_lag_6 == 0
    assert x.iloc[6].pm25_mean_6 == 3.5
    assert x.iloc[6].pm25_change_1 == 1
    assert audit.iloc[6].actual == 7
    assert x.iloc[0].history_count_6 == 1
    assert np.isnan(x.iloc[1].pm25_mean_6)
    assert x.iloc[2].pm25_mean_6 == 1
    assert np.isnan(x.iloc[10].pm25_mean_24)
    assert x.iloc[11].pm25_mean_24 == 5.5
    assert np.isnan(x.iloc[0].pm25_lag_24)
    frame, _ = load_data(*save_source(raw_hours.drop(index=11)))
    x, audit = build_features(frame, SETTINGS), build_labels(frame)
    assert audit.iloc[10].target_hour_absent and not audit.iloc[10].eligible
    assert audit.iloc[10].target_time == frame.index[11]
    assert np.isnan(audit.iloc[10].actual)
    assert np.isnan(x.iloc[12].pm25_lag_1)
    assert x.iloc[12].pm25_lag_2 == 10
    with pytest.raises(ValueError, match="complete hourly grid"):
        build_features(frame.drop(index=frame.index[11]), SETTINGS)


def test_missing_current_and_target_have_explicit_reasons(raw_hours, save_source):
    raw_hours.loc[20, "pm2.5"] = np.nan
    frame, _ = load_data(*save_source(raw_hours))
    audit = build_labels(frame)
    assert audit.iloc[20].current_pm25_missing and not audit.iloc[20].eligible
    assert audit.iloc[19].target_pm25_missing and not audit.iloc[19].eligible
    assert not audit.iloc[-1].eligible and audit.iloc[-1].target_hour_absent


def test_future_mutation_preserves_inputs_and_forecast_but_changes_label(small_config):
    frame, partitions, _ = prepare(small_config)
    train_x, train_audit = partitions["train"]
    model = make_model("ridge", {"alpha": 1}, train_x, small_config)
    model.fit(train_x, train_audit.actual)
    origin = frame.index[200]
    x = build_features(frame, small_config["features"])
    changed = frame.copy()
    changed.loc[changed.index > origin, ["pm25", "TEMP", "DEWP", "PRES"]] = 99999
    changed.loc[changed.index > origin, "cbwd"] = "future-only"
    changed_x = build_features(changed, small_config["features"])
    pd.testing.assert_frame_equal(x.loc[:origin], changed_x.loc[:origin])
    np.testing.assert_allclose(forecast(model, x.loc[[origin]])[1],
                               forecast(model, changed_x.loc[[origin]])[1])
    assert build_labels(frame).loc[origin, "actual"] != build_labels(changed).loc[origin, "actual"]
    # Truncate at t: inference still works without a future target.
    truncated = build_features(frame.loc[:origin], small_config["features"])
    np.testing.assert_allclose(forecast(model, truncated.iloc[[-1]])[1],
                               forecast(model, x.loc[[origin]])[1])


@pytest.mark.parametrize("start", ["2012-12-31 23:00", "2014-01-31 23:00", "2020-02-28 23:00"])
def test_calendar_boundary(raw_hours, save_source, start):
    time = pd.date_range(start, periods=len(raw_hours), freq="h")
    for column in ["year", "month", "day", "hour"]:
        raw_hours[column] = getattr(time, column)
    frame, _ = load_data(*save_source(raw_hours))
    audit, x = build_labels(frame), build_features(frame, SETTINGS)
    assert audit.iloc[0].target_time == pd.Timestamp(start, tz="Asia/Shanghai") + pd.Timedelta(hours=1)
    assert x.iloc[0].target_hour_sin == pytest.approx(0)
    assert x.iloc[0].target_hour_cos == pytest.approx(1)
    assert x.iloc[0].target_day_of_week == audit.iloc[0].target_time.dayofweek
    assert x.iloc[0].target_month_sin == pytest.approx(np.sin(2 * np.pi * (time[1].month - 1) / 12))


def test_target_partition_boundary_and_coverage(raw_hours, save_source):
    frame, _ = load_data(*save_source(raw_hours))
    x, audit = build_features(frame, SETTINGS), build_labels(frame)
    splits = {"train": ["2012-12-31", "2013-01-01"],
              "validation": ["2013-01-01", "2013-01-02"],
              "test": ["2013-01-02", "2013-01-03"]}
    parts, coverage = split_data(x, audit, splits, "Asia/Shanghai")
    boundary = pd.Timestamp("2012-12-31 23:00", tz="Asia/Shanghai")
    assert boundary not in parts["train"][0].index
    assert boundary in parts["validation"][0].index
    assert pd.Timestamp("2013-01-01 23:00", tz="Asia/Shanghai") in parts["test"][0].index
    assert coverage["train"]["eligible_rows"] == 23
    assert coverage["train"]["no_origin_in_source"] == 1
    assert np.isnan(parts["validation"][0].loc[boundary, "pm25_lag_24"])
    # At the next validation origin, the 24-hour lag crosses the split legitimately.
    assert parts["validation"][0].iloc[1].pm25_lag_24 == 0
