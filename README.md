# Banknote Authentication ML API

End-to-end machine learning project for banknote authentication. The model is trained in a notebook and served through a FastAPI inference endpoint.

## Features

- RandomForest-based banknote classifier
- Anti-overfitting training configuration (depth and leaf constraints)
- Serialized model artifact (`classifier.pkl`)
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
|-- Banknote.py
|-- modelTraining.ipynb
|-- BankNote_Authentication.csv
|-- classifier.pkl
|-- requirements.txt
|-- pyproject.toml
```

## Quick Start

### Option 1: Using uv (recommended)

```powershell
uv sync
uv run python app.py
```

### Option 2: Using pip + venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

API runs at `http://127.0.0.1:5001`.

## API Endpoints

- `GET /` - health/info response
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

## License

This project is licensed under the MIT License.
