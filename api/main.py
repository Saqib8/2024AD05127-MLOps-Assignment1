"""FastAPI service exposing the heart-disease classifier.

Endpoints
---------
GET  /            -> service metadata
GET  /health      -> liveness/readiness probe (model loaded?)
POST /predict     -> JSON in, prediction + confidence out
GET  /metrics     -> Prometheus metrics (request count, latency, predictions)

The service loads the persisted sklearn pipeline (preprocessing + model) once
at startup. Every request is logged and instrumented for monitoring.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# Make ``src`` importable whether run from repo root or the api/ folder.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.schema import HeartDiseaseFeatures, PredictionResponse  # noqa: E402
from src.config import FEATURE_COLUMNS, MODEL_PATH, METADATA_PATH  # noqa: E402

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("heart-disease-api")

# --------------------------------------------------------------------------- #
# Prometheus metrics
# --------------------------------------------------------------------------- #
PREDICTION_COUNTER = Counter(
    "heart_predictions_total", "Total predictions served", ["prediction"]
)
REQUEST_COUNTER = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds", "Latency of /predict in seconds"
)

# --------------------------------------------------------------------------- #
# App + model loading
# --------------------------------------------------------------------------- #
_model = None
_metadata: dict = {}


def load_model() -> None:
    global _model, _metadata
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
        if METADATA_PATH.exists():
            _metadata = json.loads(METADATA_PATH.read_text())
    else:
        logger.warning("Model file not found at %s. Train the model first.", MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    REQUEST_COUNTER.labels(endpoint="/", status="200").inc()
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "model_loaded": _model is not None,
        "best_model": _metadata.get("best_model"),
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    status = "healthy" if _model is not None else "model_not_loaded"
    REQUEST_COUNTER.labels(endpoint="/health", status="200").inc()
    return {"status": status, "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartDiseaseFeatures) -> PredictionResponse:
    if _model is None:
        REQUEST_COUNTER.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.time()
    payload = features.model_dump()
    logger.info("Prediction request: %s", payload)

    X = pd.DataFrame([payload])[FEATURE_COLUMNS]
    try:
        proba = float(_model.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNTER.labels(endpoint="/predict", status="500").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    confidence = proba if pred == 1 else 1 - proba
    response = PredictionResponse(
        prediction=pred,
        label="Heart disease present" if pred == 1 else "No heart disease",
        confidence=round(confidence, 4),
        probability_disease=round(proba, 4),
    )

    PREDICT_LATENCY.observe(time.time() - start)
    PREDICTION_COUNTER.labels(prediction=str(pred)).inc()
    REQUEST_COUNTER.labels(endpoint="/predict", status="200").inc()
    logger.info("Prediction response: %s", response.model_dump())
    return response


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
