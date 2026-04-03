import json
import os
import pickle
from pathlib import Path
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
MODEL_CONFIG_PATH = Path("model_config.json")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

classifier = None
model_config = None


def _load_model_config() -> dict:
    if not MODEL_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Model config file not found: '{MODEL_CONFIG_PATH}'. "
            "Create model_config.json with model_name, features, and labels."
        )

    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)

    if not isinstance(loaded, dict):
        raise RuntimeError("Invalid model_config.json: root must be an object.")

    features = loaded.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError("Invalid model_config.json: 'features' must be a non-empty list.")

    loaded["features"] = [str(feature).strip().lower() for feature in features]
    loaded["model_name"] = str(loaded.get("model_name", "Unnamed Model"))

    labels = loaded.get("labels", {})
    if not isinstance(labels, dict):
        raise RuntimeError("Invalid model_config.json: 'labels' must be an object.")
    loaded["labels"] = {str(key): str(value) for key, value in labels.items()}
    return loaded


def _example_payload() -> dict:
    if model_config is None:
        return {}
    return {feature: 0.0 for feature in model_config["features"]}


def _ensure_runtime_loaded() -> None:
    global classifier, model_config

    if model_config is None:
        model_config = _load_model_config()

    if classifier is None:
        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            raise RuntimeError(
                f"Model file not found: '{model_file}'. "
                "Set MODEL_PATH to a valid pickle model file."
            )
        with model_file.open("rb") as pickle_in:
            classifier = pickle.load(pickle_in)


def _normalize_payload(payload: dict) -> dict:
    normalized = {str(key).strip().lower(): value for key, value in payload.items()}
    if "kurtosis" not in normalized and "curtosis" in normalized:
        normalized["kurtosis"] = normalized["curtosis"]
    return normalized


async def _read_payload(request: Request) -> dict:
    content_type = request.headers.get("content-type", "").lower()
    raw_body = await request.body()

    if not raw_body:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request body is empty.",
                "example": _example_payload(),
            },
        )

    if "multipart/form-data" in content_type:
        raise HTTPException(
            status_code=415,
            detail={
                "message": "Use Body -> raw -> JSON in Postman for this endpoint.",
                "example": _example_payload(),
            },
        )

    if "application/x-www-form-urlencoded" in content_type:
        parsed = parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
        payload = {key: values[-1] for key, values in parsed.items()}
    else:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail={
                    "message": "Unsupported body format. Send JSON.",
                    "example": _example_payload(),
                },
            ) from exc

    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]

    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], dict):
        payload = payload["data"]

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Payload must be a JSON object with banknote features.",
                "example": _example_payload(),
            },
        )

    return _normalize_payload(payload)


def _build_feature_vector(payload: dict) -> list[float]:
    required_features = model_config["features"]
    missing_features = [feature for feature in required_features if feature not in payload]
    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features,
            },
        )

    feature_vector = []
    invalid_features = []
    for feature in required_features:
        try:
            feature_vector.append(float(payload[feature]))
        except (TypeError, ValueError):
            invalid_features.append(feature)

    if invalid_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature values must be numeric.",
                "invalid_features": invalid_features,
            },
        )

    return feature_vector


@app.on_event("startup")
def startup_event() -> None:
    _ensure_runtime_loaded()


@app.get('/')
def home():
    _ensure_runtime_loaded()
    return {
        "model": model_config["model_name"],
        "expected_features": model_config["features"],
        "example_payload": _example_payload(),
    }


@app.get('/config')
def get_config():
    _ensure_runtime_loaded()
    return model_config


@app.post('/')
def post_root_hint():
    _ensure_runtime_loaded()
    return {
        "message": "Use POST /predict for predictions.",
        "example": _example_payload(),
    }


@app.post('/predict')
async def predict_banknote(request: Request):
    _ensure_runtime_loaded()
    payload = await _read_payload(request)
    feature_vector = _build_feature_vector(payload)

    prediction_raw = int(classifier.predict([feature_vector])[0])
    prediction_label = model_config["labels"].get(str(prediction_raw), str(prediction_raw))

    return {
        "prediction": prediction_label,
        "prediction_raw": prediction_raw,
        "model": model_config["model_name"],
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)