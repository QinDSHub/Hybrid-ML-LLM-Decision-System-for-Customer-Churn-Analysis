FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1         MODEL_BUNDLE_PATH=/app/artifacts/model/model_bundle.joblib

WORKDIR /app

RUN apt-get update         && apt-get install -y --no-install-recommends build-essential         && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY azure ./azure
COPY examples ./examples
COPY artifacts ./artifacts
COPY .env.example ./
COPY README.md ./
COPY Makefile ./

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
