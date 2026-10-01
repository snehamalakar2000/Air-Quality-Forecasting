import copy

import numpy as np
import pandas as pd
import pytest

from air_quality.evaluation import metrics, error_slices, prediction_table
from air_quality.features import build_features
from air_quality.models import make_model, predict, persistence, forecast
from air_quality.pipeline import prepare


def test_persistence_and_metric_arithmetic():
    inputs = pd.DataFrame({"pm25_current": [10., 20.]})
    np.testing.assert_array_equal(persistence(inputs), [10, 20])
    result = metrics([12, 16], persistence(inputs))
    assert result["mae"] == 3
    assert result["rmse"] == pytest.approx(np.sqrt(10))
    assert result["mean_signed_error"] == 1
    assert result["fraction_underpredicted"] == 0.5
    assert metrics([], [])["mae"] is None
    with pytest.raises(ValueError, match="observed current"):
        persistence(pd.DataFrame({"pm25_current": [np.nan]}))
    with pytest.raises(ValueError, match="finite"):
        metrics([np.nan], [1])
    with pytest.raises(ValueError, match="matching shapes"):
        metrics([1, 2], [1])


@pytest.mark.parametrize("kind,settings", [("ridge", {"alpha": 1}), ("forest", {"max_depth": 3})])
def test_training_median_flags_unknown_wind_and_frozen_preprocessing(small_config, kind, settings):
    frame, parts, _ = prepare(small_config)
    train_x, train_audit = parts["train"]
    model = make_model(kind, settings, train_x, small_config)
    model.fit(train_x, train_audit.actual)
    preprocessing = model.named_steps["preprocessing"]
    numeric = preprocessing.named_transformers_["numeric"]
    columns = preprocessing.transformers_[0][2]
    column = columns.index("TEMP")
    stats = numeric.named_steps["imputer"].statistics_.copy()
    assert stats[column] == pytest.approx(train_x.TEMP.median())
    missing = train_x.loc[train_x.TEMP.isna()]
    assert len(missing) == 1 and missing.iloc[0].TEMP_missing == 1
    imputed = numeric.named_steps["imputer"].transform(missing[columns])
    assert imputed[0, column] == stats[column]
    before_shape = preprocessing.transform(train_x.iloc[[0]]).shape
    categories = copy.deepcopy(preprocessing.named_transformers_["wind"].categories_)
    before_regressor = copy.deepcopy(model.named_steps["regressor"])
    scaler_mean = numeric.named_steps["scaler"].mean_.copy() if kind == "ridge" else None
    extreme = parts["test"][0].copy()
    extreme["TEMP"] = 100000
    extreme["cbwd"] = "unseen-category"
    assert preprocessing.transform(extreme).shape[1] == before_shape[1]
    assert np.isfinite(predict(model, extreme)).all()
    np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_, stats)
    for a, b in zip(preprocessing.named_transformers_["wind"].categories_, categories):
        np.testing.assert_array_equal(a, b)
    if kind == "ridge":
        np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, scaler_mean)
        np.testing.assert_array_equal(model.named_steps["regressor"].coef_, before_regressor.coef_)
    else:
        for a, b in zip(model.named_steps["regressor"].estimators_, before_regressor.estimators_):
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
            np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
    no_label_inputs = build_features(frame.iloc[:210], small_config["features"])
    times, predictions = forecast(model, no_label_inputs)
    assert len(times) == no_label_inputs.pm25_current.notna().sum()
    assert (predictions >= 0).all()


def test_entire_training_feature_missing_and_empty_training(small_config):
    _, parts, _ = prepare(small_config)
    train_x = parts["train"][0].copy()
    train_x["DEWP"] = np.nan
    with pytest.raises(ValueError, match="Entire training numeric feature missing"):
        make_model("ridge", {"alpha": 1}, train_x, small_config)
    with pytest.raises(ValueError, match="Empty eligible training"):
        make_model("ridge", {"alpha": 1}, train_x.iloc[:0], small_config)


def test_common_population_and_empty_high_slice(small_config):
    _, parts, _ = prepare(small_config)
    train_x, train_audit = parts["train"]
    models = {}
    for kind, settings in [("ridge", {"alpha": 1}), ("forest", {"max_depth": 3})]:
        models[kind] = make_model(kind, settings, train_x, small_config).fit(train_x, train_audit.actual)
    test_x, test_audit = parts["test"]
    table = prediction_table(test_x, test_audit, models)
    assert len(table) == len(test_x) == table.target_time.nunique()
    assert table[["persistence", "ridge", "forest"]].notna().all().all()
    assert (table.target_time - table.origin_time).eq(pd.Timedelta(hours=1)).all()
    slices = error_slices(table, threshold=1000000)
    for kind in models:
        assert slices["high_pollution"][kind] == {"count": 0, "mae": None, "rmse": None,
                                                  "mean_signed_error": None, "fraction_underpredicted": None}
    with pytest.raises(ValueError, match="misaligned"):
        prediction_table(test_x.iloc[::-1], test_audit, models)


def test_nonnegative_prediction_rule():
    class NegativePredictor:
        def predict(self, inputs):
            return np.array([-2, 0, 5000])
    np.testing.assert_array_equal(predict(NegativePredictor(), None), [0, 0, 5000])
