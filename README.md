# Banknote Authentication ML API

End-to-end machine learning project for banknote authentication. The model is trained in a notebook and served through a FastAPI inference endpoint.

## Features

- Model-agnostic inference API (schema-driven)
- Anti-overfitting training configuration (depth and leaf constraints)
- External model config (`model_config.json`) for features and labels
- Serialized model artifact loaded from `MODEL_PATH` (default: `model.pkl`)
- FastAPI prediction endpoint (`/predict`)
- Postman-ready payload handling (JSON and `x-www-form-urlencoded`)

## Tech Stack

- Python 3.12+
- scikit-learn
- pandas / numpy
- FastAPI + Uvicorn

## Project Structure

```text
.
|-- app.py
|-- modelTraining.ipynb
|-- BankNote_Authentication.csv
|-- model.pkl (or set MODEL_PATH)
|-- model_config.json
|-- requirements.txt
|-- pyproject.toml
```

## Quick Start

### Option 1: Using uv (recommended)

```powershell
uv sync
uv run python app.py
```

If your model file has another name, set `MODEL_PATH`.

```powershell
$env:MODEL_PATH = "classifier.pkl"
uv run python app.py
```

### Option 2: Using pip + venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH = "classifier.pkl"
python app.py
```

API runs at `http://127.0.0.1:5001`.

## API Endpoints

- `GET /` - health/info response
- `GET /config` - loaded model config
- `POST /` - usage hint
- `POST /predict` - predict banknote class
- `GET /docs` - Swagger UI

## Postman Test

1. Method: `POST`
2. URL: `http://127.0.0.1:5001/predict`
3. Body: `raw`
4. Type: `JSON`
5. Payload:

```json
{
  "variance": 3.6216,
  "skewness": 8.6661,
  "kurtosis": -2.8073,
  "entropy": -0.44699
}
```

Compatibility note: `curtosis` is also accepted as an alias for older payloads.

Prediction response format:

```json
{
  "prediction": "Genuine",
  "prediction_raw": 0,
  "model": "Banknote Authenticator"
}
```

## cURL Example

```bash
curl -X POST "http://127.0.0.1:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{"variance":3.6216,"skewness":8.6661,"kurtosis":-2.8073,"entropy":-0.44699}'
```

## Retraining the Model

1. Open `modelTraining.ipynb`
2. Run all cells to train and evaluate
3. Re-export `classifier.pkl`

## Troubleshooting

- `422 Unprocessable Entity`: ensure Postman Body is `raw` + `JSON`
- `Address already in use`: stop existing process on port `5001`
- If payload key typo exists, `curtosis` is supported as fallback
- If startup fails with model file error, set `MODEL_PATH` to your pickle file

## Deployment

### GitHub Secrets

Set the following repository secrets before using the deployment workflow:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

- Docker image is auto-built and pushed to DockerHub on every push to `main`
- Render deploys automatically by watching the `main` branch
- Live API: `<your-render-url-here>`

### Render Setup (one-time)

1. Go to render.com -> New -> Web Service
1. Connect this GitHub repo
1. Build Command: `pip install -r requirements.txt`
1. Start Command: `uvicorn app:app --host 0.0.0.0 --port 5001`
1. Add environment variable: `MODEL_PATH=classifier.pkl`
1. Deploy

### Run Locally with Docker

```powershell
docker build -t banknote-auth-ml-api:local .
docker run --rm -p 5001:5001 -e MODEL_PATH=classifier.pkl banknote-auth-ml-api:local
```

Tech debt: `classifier.pkl` is still committed in the repository for convenience. Move model artifacts to object storage or a model registry in production.

## License

This project is licensed under the MIT License.
