"""Run from the project root: python scripts/diagnose_ridge.py.

Print local numerical-library details, inspect inputs, then exercise the failing
operation and Ridge. Runtime warnings are errors; failed probes are reported and
make this script exit with status 1. No project artifacts are modified.
"""

import json
import platform
import sys
import tempfile
import warnings

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info

from air_quality.models import make_model, predict
from air_quality.pipeline import prepare, read_config, synthetic_fixture


def probe(name, function):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with np.errstate(divide="raise", over="raise", invalid="raise"):
                function()
        print(f"PASS: {name}", flush=True)
        return True
    except (RuntimeWarning, FloatingPointError, ValueError, AssertionError) as error:
        print(f"FAIL: {name}: {type(error).__name__}: {error}", flush=True)
        return False


def inspect_inputs(config, label):
    _, partitions, _ = prepare(config)
    x, audit = partitions["train"]
    model = make_model("ridge", {"alpha": 1.0}, x, config)
    z = model.named_steps["preprocessing"].fit_transform(x)
    y = audit.actual.to_numpy(dtype=float)
    print(f"{label}: shape={z.shape}, dtype={z.dtype}, "
          f"inputs_finite={np.isfinite(z).all()}, targets_finite={np.isfinite(y).all()}, "
          f"max_abs={np.max(np.abs(z)):.8f}", flush=True)
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite transformed values: investigate data/preprocessing first")
    centered = z - z.mean(axis=0)

    def gram_check():
        reference = np.einsum("ki,kj->ij", centered, centered, optimize=False)
        gram = centered.T @ centered
        assert np.isfinite(gram).all()
        np.testing.assert_allclose(gram, reference, rtol=1e-12, atol=1e-9)

    def fit_check():
        model.fit(x, audit.actual)
        values = predict(model, partitions["validation"][0])
        assert np.isfinite(values).all()
        print(f"{label}: fitted solver={model.named_steps['regressor'].solver_}", flush=True)

    gram_ok = probe(f"{label} centered X.T @ X agrees with direct sums", gram_check)
    ridge_ok = probe(f"{label} Ridge fit and validation prediction", fit_check)
    return gram_ok and ridge_ok


def main():
    print("Python:", sys.version, "\nExecutable:", sys.executable)
    print("Platform:", platform.platform(), "\nMachine:", platform.machine())
    print("NumPy:", np.__version__, "SciPy:", scipy.__version__, "sklearn:", sklearn.__version__)
    print("NumPy build configuration (look for blas / Accelerate / OpenBLAS):")
    print(json.dumps(np.show_config(mode="dicts"), indent=2, default=str))
    print("Detected thread pools (may omit Accelerate):", threadpool_info(), flush=True)

    def identity_check():
        identity = np.eye(15)
        np.testing.assert_array_equal(identity @ identity, identity)

    outcomes = [probe("identity matrix multiplication independent of project data", identity_check)]
    with tempfile.TemporaryDirectory(prefix="ridge-diagnostic-") as directory:
        outcomes.append(probe("synthetic input inspection and numerical checks",
                              lambda: require_success(inspect_inputs(synthetic_fixture(directory), "synthetic"))))
    outcomes.append(probe("real input inspection and numerical checks",
                          lambda: require_success(inspect_inputs(read_config("config.json"), "real"))))
    return 0 if all(outcomes) else 1


def require_success(success):
    if not success:
        raise AssertionError("One or more numerical probes failed; see FAIL lines above")


if __name__ == "__main__":
    raise SystemExit(main())
