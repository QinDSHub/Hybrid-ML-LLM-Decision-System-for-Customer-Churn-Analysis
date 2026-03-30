PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) scripts/train_pipeline.py --raw-dir data/raw --artifact-dir artifacts/model

demo-model:
	$(PYTHON) scripts/bootstrap_demo_model.py

api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker-build:
	docker build -t churn-prediction-api:local .
