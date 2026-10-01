"""Separate validation selection from frozen, non-fitting test evaluation."""

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd

from .data import load_data, sha256
from .evaluation import error_slices, metrics, plot_reports, prediction_table
from .features import build_features, build_labels, split_data
from .models import make_model, persistence, predict, check_training


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if config["timezone"] != "Asia/Shanghai":
        raise ValueError("Version 1 requires Asia/Shanghai timestamps")
    previous_end = None
    for name in ["train", "validation", "test"]:
        start, end = [pd.Timestamp(t, tz=config["timezone"]) for t in config["splits"][name]]
        if start >= end or (previous_end is not None and start != previous_end):
            raise ValueError("Splits must be ordered, contiguous half-open intervals")
        if start != start.floor("h") or end != end.floor("h"):
            raise ValueError("Split boundaries must be exact hours")
        previous_end = end
    for lag in config["features"]["lags"]:
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("Lags must be positive integer hours")
    for spec in config["features"]["rolling"]:
        if not 1 <= spec["min_observed"] <= spec["hours"]:
            raise ValueError("Invalid rolling minimum")
    if config["high_pollution_quantile"] != 0.9:
        raise ValueError("Version 1 defines high pollution using the training 90th percentile")
    start, end = [pd.Timestamp(t, tz=config["timezone"]) for t in config["fixed_test_period"]]
    if start >= end:
        raise ValueError("Fixed reporting period must be nonempty")


def prepare(config):
    validate_config(config)
    frame, quality = load_data(config["data_path"], config["source"], config["timezone"])
    inputs = build_features(frame, config["features"])
    audit = build_labels(frame)
    partitions, coverage = split_data(inputs, audit, config["splits"], config["timezone"])
    quality["partitions"] = coverage
    quality["eligible_input_missingness_by_partition"] = {
        name: {column: int(x[column].isna().sum()) for column in x}
        for name, (x, _) in partitions.items()}
    reasons = ["current_row_absent", "current_pm25_missing", "target_hour_absent",
               "target_pm25_missing"]
    quality["all_source_origins"] = {
        "count": len(audit), "eligible_with_observed_next_hour": int(audit.eligible.sum()),
        "total_unique_excluded_origins": int((~audit.eligible).sum()),
        "exclusions_by_reason": {reason: int(audit[reason].sum()) for reason in reasons},
        "note": "Reasons overlap; includes the final origin whose next hour is outside the source"}
    return frame, partitions, quality


def validate(config):
    _, partitions, quality = prepare(config)
    check_training(partitions["train"][0])
    for name in ["validation", "test"]:
        if partitions[name][0].empty:
            raise ValueError(f"Empty eligible {name} split")
    write_json(Path(config["output_dir"]) / "data_quality.json", quality)
    return quality


def metadata(config, quality):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "pytest"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "seed": config["seed"], "effective_configuration": config,
            "configuration_sha256": config_hash(config), "source_sha256": quality["sha256"],
            "eligible_counts": {k: v["eligible_rows"] for k, v in quality["partitions"].items()},
            "splits": config["splits"], "git_commit": commit,
            "git_commit_note": "null means no accessible repository commit",
            "mode": "rolling one-hour; training-only fit; no refit on validation"}


def select(config):
    _, partitions, quality = prepare(config)
    train_x, train_audit = partitions["train"]
    val_x, val_audit = partitions["validation"]
    check_training(train_x)
    if val_x.empty:
        raise ValueError("Empty eligible validation split")
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    best = {}
    baseline = metrics(val_audit.actual, persistence(val_x))
    rows.append({"model": "persistence", "parameters": "{}", **baseline})
    grids = {"ridge": [{"alpha": a} for a in config["models"]["ridge_alphas"]],
             "forest": [{"max_depth": d} for d in config["models"]["forest_depths"]]}
    if not all(grids.values()):
        raise ValueError("Both model grids must be nonempty")
    for kind, grid in grids.items():
        for settings in grid:
            model = make_model(kind, settings, train_x, config)
            model.fit(train_x, train_audit.actual)
            scores = metrics(val_audit.actual, predict(model, val_x))
            rows.append({"model": kind, "parameters": json.dumps(settings, sort_keys=True), **scores})
            key = (scores["mae"], scores["rmse"])
            if kind not in best or key < best[kind]["key"]:
                best[kind] = {"key": key, "settings": settings, "model": model, "metrics": scores}
    selected = min(best, key=lambda kind: (*best[kind]["key"], 0 if kind == "ridge" else 1))
    recommended = "persistence" if baseline["mae"] <= best[selected]["metrics"]["mae"] else selected
    manifest = {"configuration_sha256": config_hash(config), "source_sha256": quality["sha256"],
                "selected_learned_model": selected, "recommended_method": recommended,
                "selection_rule": "validation MAE, then RMSE, then Ridge; persistence wins equal MAE",
                "high_pollution_threshold": float(train_audit.actual.quantile(0.9)),
                "threshold_source": "eligible training targets only, quantile=0.9",
                "fitted_on": config["splits"]["train"], "models": {},
                "persistence_validation_metrics": baseline}
    for kind in best:
        path = directory / f"{kind}.joblib"
        joblib.dump(best[kind]["model"], path)
        manifest["models"][kind] = {"path": path.name, "sha256": sha256(path),
                                    "parameters": best[kind]["settings"],
                                    "validation_metrics": best[kind]["metrics"]}
    joblib.dump(best[selected]["model"], directory / "model.joblib")
    manifest["selected_pipeline"] = {"path": "model.joblib", "sha256": sha256(directory / "model.joblib")}
    # Freeze every selection decision before evaluating the held-out test year.
    pd.DataFrame(rows).to_csv(directory / "validation_metrics.csv", index=False)
    write_json(directory / "data_quality.json", quality)
    write_json(directory / "selected_config.json", manifest)
    run = metadata(config, quality)
    run.update({"stage": "selection", "selected_parameters": {k: v["settings"] for k, v in best.items()},
                "selected_learned_model": selected, "recommended_method": recommended})
    write_json(directory / "run_metadata.json", run)
    return manifest


def load_selection(config, selection_path):
    selection_path = Path(selection_path)
    manifest = json.loads(selection_path.read_text())
    if manifest["configuration_sha256"] != config_hash(config):
        raise ValueError("Selection configuration mismatch; evaluation cannot tune or refit")
    if manifest["source_sha256"] != sha256(config["data_path"]):
        raise ValueError("Selection dataset mismatch")
    records = {**manifest["models"], "selected_pipeline": manifest["selected_pipeline"]}
    for record in records.values():
        path = selection_path.parent / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Fitted model artifact checksum mismatch")
    # Only load trusted artifacts generated by this project; joblib is not a safe interchange format.
    models = {kind: joblib.load(selection_path.parent / record["path"])
              for kind, record in manifest["models"].items()}
    return manifest, models


def evaluate(config, selection_path):
    manifest, models = load_selection(config, selection_path)
    frame, partitions, quality = prepare(config)
    inputs, audit = partitions["test"]
    if inputs.empty:
        raise ValueError("Empty eligible test split")
    table = prediction_table(inputs, audit, models)
    slices = error_slices(table, manifest["high_pollution_threshold"])
    baseline_mae = slices["overall"]["persistence"]["mae"]
    improvements = {kind: (1 - slices["overall"][kind]["mae"] / baseline_mae
                           if baseline_mae else None) for kind in models}
    report = {"selected_learned_model": manifest["selected_learned_model"],
              "recommended_method_from_validation": manifest["recommended_method"],
              "high_pollution_threshold": manifest["high_pollution_threshold"],
              "high_pollution_definition": manifest["threshold_source"],
              "slices": slices, "relative_mae_improvement_over_persistence": improvements,
              "test_coverage": quality["partitions"]["test"],
              "note": "Test results are reporting only; no fitting or selection occurred"}
    directory = Path(config["output_dir"])
    table.to_csv(directory / "test_predictions.csv", index=False)
    write_json(directory / "test_metrics.json", report)
    write_json(directory / "data_quality.json", quality)
    run = metadata(config, quality)
    run.update({"stage": "evaluation", "selection_sha256": sha256(selection_path),
                "selected_parameters": {k: v["parameters"] for k, v in manifest["models"].items()},
                "selected_learned_model": manifest["selected_learned_model"],
                "recommended_method": manifest["recommended_method"]})
    write_json(directory / "run_metadata.json", run)
    plot_reports(frame, table, slices, manifest["high_pollution_threshold"],
                 manifest["selected_learned_model"], config, directory / "plots")
    return report


def synthetic_fixture(directory):
    """A small documented source used only by tests/offline smoke, never real selection."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    time = pd.date_range("2020-02-27", periods=240, freq="h")
    x = np.arange(len(time))
    raw = pd.DataFrame({"No": x + 1, "year": time.year, "month": time.month,
                        "day": time.day, "hour": time.hour,
                        "pm2.5": 30 + 12 * np.sin(x / 8) + x / 15,
                        "DEWP": x % 7 - 4, "TEMP": x % 24 - 5, "PRES": 1010 + x % 5,
                        "cbwd": np.where(x % 2, "NW", "NE"), "Iws": 1 + x % 15,
                        "Is": np.zeros(len(x)), "Ir": np.zeros(len(x))})
    raw.loc[35, "pm2.5"] = np.nan
    raw.loc[44, "TEMP"] = np.nan
    raw = raw.drop(index=50)
    path = directory / "synthetic.csv"
    raw.to_csv(path, index=False, na_rep="NA")
    config = read_config(Path(__file__).resolve().parents[2] / "config.json")
    config = copy.deepcopy(config)
    config["data_path"] = str(path)
    config["source"] = {"sha256": sha256(path), "size_bytes": path.stat().st_size,
                        "rows": len(raw), "start": str(time.min()), "end": str(time.max())}
    config["splits"] = {"train": [str(time[0]), str(time[144])],
                        "validation": [str(time[144]), str(time[192])],
                        "test": [str(time[192]), str(time[-1] + pd.Timedelta(hours=1))]}
    config["fixed_test_period"] = config["splits"]["test"]
    config["models"].update(n_estimators=8, ridge_alphas=[0.1, 1.0], forest_depths=[3, 5])
    config["output_dir"] = str(directory / "artifacts")
    return config


def verify_outputs(config):
    directory = Path(config["output_dir"])
    required = ["data_quality.json", "validation_metrics.csv", "selected_config.json",
                "model.joblib", "ridge.joblib", "forest.joblib", "run_metadata.json",
                "test_metrics.json", "test_predictions.csv"]
    for name in required:
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise ValueError(f"Missing smoke output: {name}")
    figures = {path.name for path in (directory / "plots").glob("*.png")}
    if figures != {"training_overview.png", "test_predictions_fixed_period.png",
                   "model_error_comparison.png"}:
        raise ValueError("Smoke must produce exactly the three required figures")
    table = pd.read_csv(directory / "test_predictions.csv")
    for kind in ["persistence", "ridge", "forest"]:
        if not np.isfinite(table[kind]).all() or (table[kind] < 0).any():
            raise ValueError("Invalid smoke predictions")
    target, origin = pd.to_datetime(table.target_time), pd.to_datetime(table.origin_time)
    if not ((target - origin) == pd.Timedelta(hours=1)).all():
        raise ValueError("Smoke forecasts are not exact-hour")


def smoke():
    with tempfile.TemporaryDirectory(prefix="air-quality-smoke-") as directory:
        config = synthetic_fixture(directory)
        validate(config)
        select(config)
        report = evaluate(config, Path(config["output_dir"]) / "selected_config.json")
        verify_outputs(config)
        return {"status": "passed", "synthetic_test_rows": report["slices"]["overall"]["ridge"]["count"]}
