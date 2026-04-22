"""
Tests for fingerprint-based models using synthetic dummy data.
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.fingerprint_models import (
    LogisticRegressionBBB,
    SVMBBB,
    MLPBBB,
)

#Synthetic ECFP4-like data
np.random.seed(0)
N_SAMPLES = 200
N_FEATURES = 2048
X_DUMMY = np.random.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
y_DUMMY = np.random.randint(0, 2, N_SAMPLES)

# train / val split
X_train, X_val = X_DUMMY[:160], X_DUMMY[160:]
y_train, y_val = y_DUMMY[:160], y_DUMMY[160:]

ALL_MODELS = [LogisticRegressionBBB, SVMBBB, MLPBBB]


#  Basic interface tests
@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_fit_runs_without_error(ModelClass):
    """fit() should complete without raising."""
    model = ModelClass(seed=42)
    model.fit(X_train, y_train, X_val, y_val)


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_predict_shape_and_dtype(ModelClass):
    """predict() must return (N,) with binary int values."""
    model = ModelClass(seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == (40,), f"Expected (40,), got {preds.shape}"
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary 0/1"


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_predict_proba_shape(ModelClass):
    """predict_proba() must return (N, 2) with rows summing to 1."""
    model = ModelClass(seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    assert proba.shape == (40, 2), f"Expected (40, 2), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), \
        "Each row of predict_proba must sum to 1"


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_proba_in_valid_range(ModelClass):
    """All probabilities should be in [0, 1]."""
    model = ModelClass(seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        "Probabilities must be between 0 and 1"


#  Seed isolation
@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_different_seeds_produce_models(ModelClass):
    """Two models with different seeds should both work independently."""
    m1 = ModelClass(seed=42)
    m2 = ModelClass(seed=123)
    m1.fit(X_train, y_train, X_val, y_val)
    m2.fit(X_train, y_train, X_val, y_val)
    p1 = m1.predict(X_val)
    p2 = m2.predict(X_val)
    assert p1.shape == p2.shape == (40,)


#  predict is consistent with predict_proba
@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_predict_matches_proba(ModelClass):
    """predict() should agree with argmax of predict_proba()."""
    model = ModelClass(seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    proba_preds = (proba[:, 1] >= 0.5).astype(int)
    assert np.array_equal(preds, proba_preds), \
        "predict() and predict_proba() must be consistent"
