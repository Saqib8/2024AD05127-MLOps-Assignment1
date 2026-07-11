"""Data loading, cleaning and preprocessing pipeline.

The preprocessing is expressed as an sklearn ``ColumnTransformer`` so that the
exact same transformations used in training are reused, unmodified, at
inference time (loaded from the persisted model pipeline).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, coercing '?' placeholders to real NaNs."""
    df = pd.read_csv(path, na_values=["?"])
    if list(df.columns) != COLUMN_NAMES:
        # Handle a header-less raw file gracefully.
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values=["?"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and engineer the binary target.

    Steps:
      * Ensure all feature columns are numeric.
      * Convert the 0-4 diagnosis (``num``) into a binary target
        (0 = no disease, 1 = disease present).
      * Median-impute the two columns (``ca``, ``thal``) that carry the
        original missing values so downstream code sees a complete frame.
    """
    df = df.copy()

    # Everything in this dataset is numeric once '?' is treated as NaN.
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary target: any diagnosis code > 0 means heart disease present.
    df[TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET_COLUMN])

    # Median imputation for the columns with genuine missing values.
    for col in ["ca", "thal"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def load_clean_data(save: bool = True) -> pd.DataFrame:
    """Convenience loader: raw -> cleaned frame, optionally persisted."""
    df = clean_data(load_raw_data())
    if save:
        CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_DATA_PATH, index=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside every model pipeline.

    * Numeric features  -> median impute + standard scale.
    * Categorical codes -> most-frequent impute + one-hot encode.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame):
    """Return ``(X, y)`` with X restricted to the configured feature columns."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
