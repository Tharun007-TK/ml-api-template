FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tech debt: classifier.pkl is still committed in the repo; move model artifacts to external storage.
ENV MODEL_PATH=classifier.pkl

EXPOSE 5001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
