.PHONY: install install-dev train tune api test lint docker-build docker-run clean

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

train:
	python train.py

tune:
	python -m src.tuning

api:
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

docker-build:
	docker build -t fraud-detection-api .

docker-run:
	docker compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .pytest_cache .ruff_cache
