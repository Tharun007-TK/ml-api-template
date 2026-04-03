import json
import pickle
from urllib.parse import parse_qs

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from Banknote import Banknote

app = FastAPI()
with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

EXAMPLE_PAYLOAD = {
    "variance": 3.6216,
    "skewness": 8.6661,
    "kurtosis": -2.8073,
    "entropy": -0.44699,
}


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
                "example": EXAMPLE_PAYLOAD,
            },
        )

    if "multipart/form-data" in content_type:
        raise HTTPException(
            status_code=415,
            detail={
                "message": "Use Body -> raw -> JSON in Postman for this endpoint.",
                "example": EXAMPLE_PAYLOAD,
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
                    "example": EXAMPLE_PAYLOAD,
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
                "example": EXAMPLE_PAYLOAD,
            },
        )

    return _normalize_payload(payload)


@app.get('/')
def home():
    return {
        "message": "API is running. Use POST /predict.",
        "example": EXAMPLE_PAYLOAD,
    }


@app.post('/')
def post_root_hint():
    return {
        "message": "Use POST /predict for predictions.",
        "example": EXAMPLE_PAYLOAD,
    }

@app.post('/predict')
async def predict_banknote(request: Request):
    payload = await _read_payload(request)
    try:
        data = Banknote.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid payload fields.",
                "errors": exc.errors(),
                "example": EXAMPLE_PAYLOAD,
            },
        ) from exc

    prediction = classifier.predict([[data.variance, data.skewness, data.kurtosis, data.entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5001)