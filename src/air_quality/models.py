"""Training-only preprocessing and two intentionally small model families."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def check_training(inputs):
    if inputs.empty:
        raise ValueError("Empty eligible training split")
    missing = [c for c in inputs if c != "cbwd" and inputs[c].isna().all()]
    if missing:
        raise ValueError(f"Entire training numeric feature missing: {missing}")


def make_model(kind, settings, inputs, config):
    check_training(inputs)
    numeric = [c for c in inputs if c != "cbwd" and not c.endswith("_missing")]
    flags = [c for c in inputs if c.endswith("_missing")]
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "ridge":
        steps.append(("scaler", StandardScaler()))
        estimator = Ridge(alpha=settings["alpha"])
    elif kind == "forest":
        estimator = RandomForestRegressor(
            max_depth=settings["max_depth"], n_estimators=config["models"]["n_estimators"],
            min_samples_leaf=config["models"]["min_samples_leaf"],
            n_jobs=config["models"]["n_jobs"], random_state=config["seed"])
    else:
        raise ValueError(f"Unknown model: {kind}")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("flags", "passthrough", flags),
        ("wind", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cbwd"]),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def predict(model, inputs):
    predictions = np.maximum(0.0, model.predict(inputs))
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced nonfinite predictions")
    return predictions


def forecast(model, inputs):
    """Inference needs observed current PM2.5, but no future label."""
    eligible = inputs.pm25_current.notna()
    return inputs.index[eligible], predict(model, inputs.loc[eligible]) if eligible.any() else np.array([])


def persistence(inputs):
    if inputs.pm25_current.isna().any():
        raise ValueError("Persistence requires observed current PM2.5")
    return inputs.pm25_current.to_numpy(dtype=float)
