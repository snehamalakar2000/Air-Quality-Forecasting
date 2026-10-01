"""Distinguish nonfinite features from failures in a numerical backend."""

import numpy as np

from air_quality.models import make_model
from air_quality.pipeline import prepare


def test_ridge_preprocessing_produces_finite_inputs(small_config):
    _, partitions, _ = prepare(small_config)
    x, audit = partitions["train"]
    model = make_model("ridge", {"alpha": 1.0}, x, small_config)
    preprocessing = model.named_steps["preprocessing"]
    transformed = preprocessing.fit_transform(x)
    assert np.isfinite(transformed).all()
    assert np.isfinite(audit.actual.to_numpy()).all()
    for split in ["validation", "test"]:
        assert np.isfinite(preprocessing.transform(partitions[split][0])).all()


def test_ridge_gram_matrix_matches_direct_summation(small_config):
    _, partitions, _ = prepare(small_config)
    x, _ = partitions["train"]
    model = make_model("ridge", {"alpha": 1.0}, x, small_config)
    transformed = model.named_steps["preprocessing"].fit_transform(x)
    centered = transformed - transformed.mean(axis=0)
    # optimize=False uses direct sums rather than a BLAS matrix product.
    reference = np.einsum("ki,kj->ij", centered, centered, optimize=False)
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        gram = centered.T @ centered
        identity = np.eye(15)
        np.testing.assert_array_equal(identity @ identity, identity)
    assert np.isfinite(gram).all()
    np.testing.assert_allclose(gram, reference, rtol=1e-12, atol=1e-10)
