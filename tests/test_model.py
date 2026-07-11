"""Unit tests for model candidates, training pipeline and persisted artifact."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, MODEL_PATH, METADATA_PATH
from src.data_preprocessing import build_preprocessor
from src.train import get_model_candidates


def _make_xy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.integers(30, 70, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(100, 180, n),
            "chol": rng.integers(150, 350, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(90, 200, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 4, n).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.choice([3, 6, 7], n),
        }
    )[FEATURE_COLUMNS]
    # Target loosely tied to a couple of features so models can learn something.
    y = ((X["cp"] >= 3).astype(int) + (X["oldpeak"] > 2).astype(int) >= 1).astype(int)
    return X, y


def test_candidates_present():
    cands = get_model_candidates()
    assert "logistic_regression" in cands
    assert "random_forest" in cands
    for name, (est, grid) in cands.items():
        assert hasattr(est, "fit")
        assert isinstance(grid, dict)


def test_pipeline_trains_and_predicts_valid_range():
    X, y = _make_xy()
    est = get_model_candidates()["logistic_regression"][0]
    pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", est)])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    assert set(np.unique(preds)).issubset({0, 1})
    assert ((proba >= 0) & (proba <= 1)).all()
    # Model should do better than chance on this learnable signal.
    assert (preds == y).mean() > 0.6


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact not trained yet")
def test_persisted_model_loads_and_predicts():
    import joblib

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }])[FEATURE_COLUMNS]
    pred = model.predict(sample)
    assert pred[0] in (0, 1)


@pytest.mark.skipif(not METADATA_PATH.exists(), reason="Metadata not present yet")
def test_metadata_has_expected_metrics():
    meta = json.loads(METADATA_PATH.read_text())
    assert "best_model" in meta
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in meta["metrics"]
        assert 0.0 <= meta["metrics"][key] <= 1.0
