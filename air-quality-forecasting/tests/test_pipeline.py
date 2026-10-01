import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from air_quality.data import sha256
from air_quality.evaluation import fixed_period_grid
from air_quality.pipeline import (evaluate, select, prepare, verify_outputs, read_config,
                                  validate_config, load_selection)


def assert_equivalent(first, second):
    """Threaded forest sums may differ in their final floating-point bits."""
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equivalent(first[key], second[key])
    elif isinstance(first, float):
        assert first == pytest.approx(second, rel=1e-10, abs=1e-10)
    else:
        assert first == second


def test_end_to_end_freezing_and_repeatability(small_config, monkeypatch):
    manifest = select(small_config)
    output = Path(small_config["output_dir"])
    selection = output / "selected_config.json"
    before = {name: sha256(output / name) for name in
              ["ridge.joblib", "forest.joblib", "model.joblib", "selected_config.json"]}
    train_audit = prepare(small_config)[1]["train"][1]
    assert manifest["high_pollution_threshold"] == train_audit.actual.quantile(0.9)
    # Evaluation must never call fit, even on its normal execution path.
    from sklearn.pipeline import Pipeline
    original_fit = Pipeline.fit
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Evaluation tried to fit a model")
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    first = evaluate(small_config, selection)
    verify_outputs(small_config)
    first_predictions = pd.read_csv(output / "test_predictions.csv")
    assert all(sha256(output / name) == digest for name, digest in before.items())
    assert manifest["selected_learned_model"] == first["selected_learned_model"]
    run = json.loads((output / "run_metadata.json").read_text())
    assert run["seed"] == 42 and run["source_sha256"] == small_config["source"]["sha256"]
    assert run["splits"] == small_config["splits"] and run["selected_parameters"]
    second_evaluation = evaluate(small_config, selection)
    assert_equivalent(first, second_evaluation)
    monkeypatch.setattr(Pipeline, "fit", original_fit)
    select(small_config)
    second = evaluate(small_config, selection)
    second_predictions = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_allclose(first_predictions[["ridge", "forest", "persistence"]],
                               second_predictions[["ridge", "forest", "persistence"]], atol=1e-10)
    assert_equivalent(first, second)


def test_selection_uses_validation_only_and_recommends_persistence_on_equal_mae(small_config, monkeypatch):
    from air_quality import pipeline
    # Deliberately give every learned model persistence's validation predictions.
    monkeypatch.setattr(pipeline, "predict", lambda model, inputs: inputs.pm25_current.to_numpy())
    manifest = select(small_config)
    assert manifest["selected_learned_model"] == "ridge"
    assert manifest["recommended_method"] == "persistence"
    assert manifest["models"]["ridge"]["parameters"] == {"alpha": 0.1}
    assert manifest["fitted_on"] == small_config["splits"]["train"]


def test_evaluation_rejects_modified_config_dataset_and_model(small_config):
    select(small_config)
    selection = Path(small_config["output_dir"]) / "selected_config.json"
    changed = copy.deepcopy(small_config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="configuration mismatch"):
        evaluate(changed, selection)
    path = Path(small_config["data_path"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="dataset mismatch"):
        evaluate(small_config, selection)
    path.write_bytes(original)
    model = selection.parent / "ridge.joblib"
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        load_selection(small_config, selection)


def test_empty_validation_and_test_split_clear_failures(small_config):
    raw = pd.read_csv(small_config["data_path"])
    # Remove all eligible labels from validation and test while retaining timestamps.
    raw.loc[raw.index >= 143, "pm2.5"] = np.nan
    path = Path(small_config["data_path"])
    raw.to_csv(path, index=False, na_rep="NA")
    small_config["source"].update(sha256=sha256(path), size_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="Empty eligible validation split"):
        select(small_config)


def test_empty_test_does_not_enter_selection_but_cannot_be_scored(small_config):
    path = Path(small_config["data_path"])
    raw = pd.read_csv(path)
    time = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    raw.loc[time >= pd.Timestamp(small_config["splits"]["test"][0]), "pm2.5"] = np.nan
    raw.to_csv(path, index=False, na_rep="NA")
    small_config["source"].update(sha256=sha256(path), size_bytes=path.stat().st_size)
    select(small_config)
    with pytest.raises(ValueError, match="Empty eligible test split"):
        evaluate(small_config, Path(small_config["output_dir"]) / "selected_config.json")


def test_fit_and_selection_predictions_never_use_test_rows(small_config, monkeypatch):
    from air_quality import pipeline
    _, partitions, _ = prepare(small_config)
    train_index = partitions["train"][0].index
    validation_index = partitions["validation"][0].index
    original_predict = pipeline.predict
    calls = {"fit": 0, "predict": 0}
    from sklearn.pipeline import Pipeline
    original_fit = Pipeline.fit
    def checked_fit(self, x, y=None, **kwargs):
        assert x.index.equals(train_index)
        assert y.index.equals(train_index)
        calls["fit"] += 1
        return original_fit(self, x, y, **kwargs)
    def checked_predict(model, x):
        assert x.index.equals(validation_index)
        calls["predict"] += 1
        return original_predict(model, x)
    monkeypatch.setattr(Pipeline, "fit", checked_fit)
    monkeypatch.setattr(pipeline, "predict", checked_predict)
    select(small_config)
    assert calls == {"fit": 4, "predict": 4}


def test_fixed_period_preserves_gaps_and_empty_window(small_config):
    select(small_config)
    output = Path(small_config["output_dir"])
    evaluate(small_config, output / "selected_config.json")
    table = pd.read_csv(output / "test_predictions.csv")
    table["target_time"] = pd.to_datetime(table.target_time)
    gapped = table.drop(index=5)
    grid = fixed_period_grid(gapped, small_config["fixed_test_period"], "Asia/Shanghai")
    assert pd.isna(grid.loc[table.target_time.iloc[5], "actual"])
    assert len(grid) == 48 and grid.actual.notna().sum() == 47
    empty = fixed_period_grid(table, ["2014-01-01", "2014-01-08"], "Asia/Shanghai")
    assert len(empty) == 168 and empty.actual.isna().all()
    # The plot code must annotate an empty period rather than replacing it.
    small_config["fixed_test_period"] = ["2014-01-01", "2014-01-08"]
    select(small_config)
    evaluate(small_config, output / "selected_config.json")
    verify_outputs(small_config)


def test_invalid_configuration(small_config):
    changed = copy.deepcopy(small_config)
    changed["features"]["lags"] = [-1]
    with pytest.raises(ValueError, match="positive integer"):
        validate_config(changed)
    changed = copy.deepcopy(small_config)
    changed["splits"]["validation"][0] = changed["splits"]["train"][0]
    with pytest.raises(ValueError, match="ordered, contiguous"):
        validate_config(changed)


def test_real_static_source_validates_without_training():
    from air_quality.pipeline import validate
    config = read_config("config.json")
    config["output_dir"] = str(Path(".pytest_cache") / "real-validation")
    quality = validate(config)
    assert quality["raw_rows"] == 43824
    assert quality["absent_hours"] == 0
    assert quality["raw_missing_per_column"]["pm25"] == 2067
