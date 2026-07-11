"""Unit tests for data cleaning and the preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.data_preprocessing import (
    build_preprocessor,
    clean_data,
    split_features_target,
)


@pytest.fixture
def raw_df():
    """A tiny raw-like frame including a missing value and multiclass target."""
    return pd.DataFrame(
        {
            "age": [63, 67, 41, 56],
            "sex": [1, 1, 0, 1],
            "cp": [1, 4, 2, 3],
            "trestbps": [145, 160, 130, 120],
            "chol": [233, 286, 204, 236],
            "fbs": [1, 0, 0, 0],
            "restecg": [0, 2, 2, 0],
            "thalach": [150, 108, 172, 178],
            "exang": [0, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 1.4, 0.8],
            "slope": [3, 2, 1, 1],
            "ca": [0.0, 3.0, np.nan, 0.0],   # missing value to be imputed
            "thal": [6.0, 3.0, 3.0, 3.0],
            "num": [0, 2, 0, 1],             # multiclass diagnosis
        }
    )


def test_clean_data_creates_binary_target(raw_df):
    out = clean_data(raw_df)
    assert TARGET_COLUMN in out.columns
    assert "num" not in out.columns
    assert set(out[TARGET_COLUMN].unique()).issubset({0, 1})
    # num > 0 -> 1
    assert list(out[TARGET_COLUMN]) == [0, 1, 0, 1]


def test_clean_data_imputes_missing(raw_df):
    out = clean_data(raw_df)
    assert out["ca"].isna().sum() == 0
    assert int(out.isna().sum().sum()) == 0


def test_split_features_target_shapes(raw_df):
    out = clean_data(raw_df)
    X, y = split_features_target(out)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(raw_df)
    assert TARGET_COLUMN not in X.columns


def test_preprocessor_output_is_numeric_and_finite(raw_df):
    out = clean_data(raw_df)
    X, _ = split_features_target(out)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    arr = np.asarray(transformed)
    assert arr.shape[0] == len(X)
    # One-hot expands columns, so there must be >= raw feature count.
    assert arr.shape[1] >= len(FEATURE_COLUMNS)
    assert np.isfinite(arr).all()


def test_feature_groups_partition_all_features():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)
