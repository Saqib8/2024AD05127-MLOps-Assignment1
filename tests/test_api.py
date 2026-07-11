"""Integration tests for the FastAPI service using TestClient."""
import pytest

from src.config import MODEL_PATH

fastapi_installed = True
try:
    from fastapi.testclient import TestClient
    from api.main import app
except Exception:  # noqa: BLE001
    fastapi_installed = False


VALID_PAYLOAD = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.skipif(not fastapi_installed, reason="fastapi not installed")
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


@pytest.mark.skipif(not fastapi_installed, reason="fastapi not installed")
def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"api_requests_total" in r.content


@pytest.mark.skipif(
    not (fastapi_installed and MODEL_PATH.exists()),
    reason="model not trained",
)
def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


@pytest.mark.skipif(not fastapi_installed, reason="fastapi not installed")
def test_predict_rejects_invalid_input(client):
    bad = dict(VALID_PAYLOAD)
    bad["sex"] = 5  # out of allowed range (0/1)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
