"""Unit tests for FastAPI inference endpoints and schema validation."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """TestClient fixture for FastAPI."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Test GET /health returns operational status and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data
    assert "disclaimer" in data


def test_predict_endpoint_valid_case(client):
    """Test POST /predict with valid patient measurements."""
    payload = {
        "age": 63.0,
        "sex": 1,
        "cp": 1,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0.0,
        "thal": 6.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1]
    assert data["prediction_label"] in ["Absence of Heart Disease", "Presence of Heart Disease"]
    assert 0.0 <= data["probability_presence"] <= 1.0
    assert 0.0 <= data["probability_absence"] <= 1.0
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    assert "disclaimer" in data


def test_predict_endpoint_invalid_category(client):
    """Test POST /predict rejects invalid categorical inputs (e.g., thal=99)."""
    payload = {
        "age": 60.0,
        "sex": 1,
        "cp": 1,
        "trestbps": 140.0,
        "chol": 220.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0.0,
        "thal": 99.0  # Invalid thalassemia category
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_out_of_range_numeric(client):
    """Test POST /predict rejects physiological impossibilities (e.g., age=-5)."""
    payload = {
        "age": -5.0,  # Invalid
        "sex": 1,
        "cp": 1,
        "trestbps": 140.0,
        "chol": 220.0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0.0,
        "thal": 3.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields(client):
    """Test POST /predict rejects incomplete request body."""
    payload = {
        "age": 55.0,
        "sex": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
