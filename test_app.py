import os

os.environ.setdefault("MODEL_PATH", "classifier.pkl")

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_root_health_returns_200() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "expected_features" in data
    assert "example_payload" in data


def test_predict_valid_payload_returns_prediction_key() -> None:
    payload = {
        "variance": 3.6216,
        "skewness": 8.6661,
        "kurtosis": -2.8073,
        "entropy": -0.44699,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_raw" in data
    assert "model" in data


def test_predict_missing_fields_returns_422() -> None:
    payload = {
        "variance": 3.6216,
        "skewness": 8.6661,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_body_returns_422() -> None:
    response = client.post("/predict", content="")
    assert response.status_code == 422
