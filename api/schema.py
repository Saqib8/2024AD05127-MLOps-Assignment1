"""Pydantic request/response schemas for the prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HeartDiseaseFeatures(BaseModel):
    """Input features for a single patient (Heart Disease UCI schema)."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (0-4)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression from exercise")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak exercise ST (0-3)")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored (0-4)")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia (3/6/7)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease, 0 = no disease")
    label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="P(heart disease)")
