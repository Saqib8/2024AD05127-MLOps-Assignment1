# Convenience targets. Usage: make <target>
.PHONY: install data train test lint api docker compose k8s clean

install:
	pip install -r requirements.txt

data:
	python data/download_data.py

train:
	python -m src.train

test:
	python -m pytest -v

lint:
	flake8 src api tests --max-line-length=120 --exit-zero --statistics

api:
	uvicorn api.main:app --reload --port 8000

mlflow:
	mlflow ui --backend-store-uri ./mlruns

docker:
	docker build -t heart-disease-api .

compose:
	docker compose up --build

k8s:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__ test-results.xml
