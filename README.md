# Heart Disease Prediction: End-to-End MLOps Pipeline

An end-to-end, production-style MLOps project that predicts the risk of heart
disease from patient health data (UCI Heart Disease / Cleveland dataset) and
serves it as a cloud-ready, monitored REST API.

> **Course:** Machine Learning Operations (MLOps), AIMLCZG523, Assignment 01

**Pipeline flow:**

```
UCI dataset -> EDA + cleaning -> model training (LogReg / RF / XGBoost)
   -> MLflow tracking -> .joblib pipeline artifact
   -> GitHub Actions CI (lint -> test -> train -> docker build)
   -> Docker image -> FastAPI /predict -> Kubernetes (LoadBalancer)
   -> Prometheus -> Grafana
```

## Highlights

| Stage | Tooling |
|-------|---------|
| Data & EDA | pandas, seaborn, matplotlib |
| Preprocessing | scikit-learn `Pipeline` + `ColumnTransformer` |
| Modelling | Logistic Regression, Random Forest, XGBoost + `GridSearchCV` |
| Experiment tracking | MLflow (params, metrics, plots, models) |
| Packaging | `joblib` pipeline + metadata JSON |
| API | FastAPI (`/predict`, `/health`, `/metrics`, `/docs`) |
| Testing | pytest (data, model & API tests) |
| CI/CD | GitHub Actions (lint, test, train, docker) |
| Containerization | Docker + docker-compose |
| Deployment | Kubernetes (Deployment, Service, Ingress, HPA) |
| Monitoring | Prometheus + Grafana |

## Best model results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:----------:|
| Logistic Regression (selected) | 0.885 | 0.839 | 0.929 | 0.881 | 0.966 | 0.902 |
| Random Forest | 0.885 | 0.839 | 0.929 | 0.881 | 0.955 | 0.898 |
| XGBoost | 0.902 | 0.867 | 0.929 | 0.897 | 0.931 | 0.870 |

Logistic Regression is selected as the production model (highest ROC-AUC).

## Repository structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py          # dataset acquisition (ucimlrepo + HTTP fallback)
│   ├── heart_disease_raw.csv      # raw download
│   └── heart_disease_clean.csv    # cleaned + binary target
├── notebooks/
│   ├── 01_eda.ipynb               # EDA + visualizations
│   ├── 02_training.ipynb          # features, models, tuning, MLflow, packaging
│   └── 03_inference.ipynb         # single + batch inference demo
├── src/
│   ├── config.py                  # paths, feature groups, constants
│   ├── data_preprocessing.py      # load/clean + ColumnTransformer pipeline
│   └── train.py                   # train, tune, MLflow log, persist best model
├── api/
│   ├── main.py                    # FastAPI service
│   └── schema.py                  # pydantic request/response models
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── models/                        # heart_disease_model.joblib + metadata
├── k8s/                           # deployment, service, ingress, hpa
├── monitoring/                    # prometheus.yml + grafana provisioning
├── .github/workflows/ci.yml       # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt               # full dev/training env
├── requirements-api.txt           # slim serving env (used by Dockerfile)
└── report/                        # final report + screenshots
```

## 1. Setup (clean environment)

```bash
git clone <your-repo-url>
cd heart-disease-mlops

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Acquire the dataset

```bash
python data/download_data.py
```
Downloads the Heart Disease UCI (Cleveland) dataset via the official
`ucimlrepo` package (id=45), falling back to a direct HTTP download.

## 3. Run the notebooks / training

Open the notebooks in order (`01_eda`, `02_training`, `03_inference`), or run
training headless:

```bash
python -m src.train
```
This trains all models, logs to MLflow, and saves the best pipeline to
`models/heart_disease_model.joblib`.

View experiments in MLflow:
```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

## 4. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Test `/predict`:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Response:
```json
{"prediction":0,"label":"No heart disease","confidence":0.8746,"probability_disease":0.1254}
```

## 5. Run the tests

```bash
python -m pytest -v
```

## 6. Docker

```bash
# Build & run just the API:
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api

# Or the full stack (API + Prometheus + Grafana):
docker compose up --build
#   API        -> http://localhost:8000/docs
#   Prometheus -> http://localhost:9090
#   Grafana    -> http://localhost:3000  (admin/admin)
```

## 7. Kubernetes deployment (Minikube example)

```bash
minikube start
# Build the image directly into Minikube's docker daemon:
eval $(minikube docker-env)          # Linux/Mac
# (Windows PowerShell: & minikube docker-env | Invoke-Expression)
docker build -t heart-disease-api:latest .

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
# Optional:
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml    # after: minikube addons enable ingress

kubectl get pods
minikube service heart-disease-api   # opens the LoadBalancer URL
```

For GKE/EKS/AKS: push the image to a registry, update `image:` in
`k8s/deployment.yaml`, and the `LoadBalancer` service provisions a public IP.

## 8. Monitoring

The API exposes Prometheus metrics at `/metrics`:
`api_requests_total`, `heart_predictions_total`, `predict_latency_seconds`.
`docker compose up` starts Prometheus (scraping the API) and Grafana with a
pre-provisioned dashboard (`Heart Disease API Monitoring`).

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service metadata |
| GET | `/health` | Liveness/readiness (model loaded?) |
| POST | `/predict` | Predict from JSON features, returns prediction + confidence |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Interactive Swagger UI |

## Reproducibility notes

* Preprocessing is embedded in the persisted sklearn `Pipeline`, so inference is
  a single artifact and matches training exactly.
* `RANDOM_STATE=42` fixes all splits and models.
* Pinned dependencies in `requirements.txt` / `requirements-api.txt`.
* The Docker image runs as a non-root user with a `/health` HEALTHCHECK.
