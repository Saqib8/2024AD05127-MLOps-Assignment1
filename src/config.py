"""Central configuration for the Heart Disease MLOps project.

Keeping paths and feature definitions in one place makes the notebooks,
training scripts, API and tests reference a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"

MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------- #
# Dataset definition (Heart Disease UCI - Cleveland)
# --------------------------------------------------------------------------- #
# Raw column names as they appear in the UCI processed.cleveland.data file.
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1/0)
    "restecg",   # resting electrocardiographic results (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise induced angina (1/0)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of the peak exercise ST segment (1-3)
    "ca",        # number of major vessels (0-3) colored by fluoroscopy
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # diagnosis of heart disease (0-4) -> raw target
]

TARGET_COLUMN = "target"          # engineered binary target
RAW_TARGET_COLUMN = "num"         # original multiclass diagnosis column

# Feature groups drive the preprocessing ColumnTransformer.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20

MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"
MLFLOW_TRACKING_URI = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
