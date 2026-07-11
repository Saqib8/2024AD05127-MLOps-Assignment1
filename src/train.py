"""Train, evaluate, track (MLflow) and persist the heart-disease classifier.

Running this module end-to-end:
  * loads and cleans the data,
  * trains Logistic Regression and Random Forest (+ optional XGBoost),
  * tunes each with GridSearchCV / cross-validation,
  * logs params, metrics, plots and models to MLflow,
  * selects the best model by ROC-AUC and persists it with metadata.

It is deliberately import-safe so the training notebook and the CI pipeline
can both call :func:`run_training`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# Allow MLflow's simple file-based tracking store (./mlruns) on newer MLflow
# versions that otherwise raise in "maintenance mode". This keeps the familiar
# ``mlflow ui`` workflow working without a database backend.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    MODELS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURE_COLUMNS,
)
from src.data_preprocessing import build_preprocessor, load_clean_data, split_features_target


def get_model_candidates() -> dict:
    """Return {name: (estimator, param_grid)} for the models we tune."""
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    }
    # XGBoost is optional; include it only if installed.
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    except ImportError:
        pass
    return candidates


def _evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def run_training(use_mlflow: bool = True, verbose: bool = True) -> dict:
    """Train all candidates, log to MLflow, persist the best model.

    Returns a summary dict with per-model metrics and the winning model name.
    """
    df = load_clean_data(save=True)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            _mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            mlflow = _mlflow
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[train] MLflow disabled: {exc}")
            mlflow = None

    results: dict[str, dict] = {}
    best_name, best_auc, best_pipeline, best_params = None, -1.0, None, None

    for name, (estimator, grid) in get_model_candidates().items():
        if verbose:
            print(f"\n[train] === {name} ===")
        pipe = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        # n_jobs=1: dataset is tiny, and it avoids a joblib/loky temp-folder
        # teardown warning on Windows when run headless (e.g. via nbconvert/CI).
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        metrics = _evaluate(best_pipe, X_test, y_test)
        cv_auc = float(
            cross_val_score(best_pipe, X_train, y_train, scoring="roc_auc", cv=cv).mean()
        )
        metrics["cv_roc_auc"] = cv_auc
        results[name] = {"metrics": metrics, "best_params": search.best_params_}

        if verbose:
            print(f"[train] best params: {search.best_params_}")
            print(f"[train] test metrics: {metrics}")

        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_type", name)
                for k, v in search.best_params_.items():
                    mlflow.log_param(k, v)
                mlflow.log_metrics(metrics)
                _log_plots_to_mlflow(mlflow, best_pipe, X_test, y_test, name)
                try:
                    mlflow.sklearn.log_model(best_pipe, name="model")
                except Exception:  # noqa: BLE001
                    pass

        if metrics["roc_auc"] > best_auc:
            best_name, best_auc = name, metrics["roc_auc"]
            best_pipeline, best_params = best_pipe, search.best_params_

    # Persist the winning pipeline (preprocessing + model together).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "best_model": best_name,
        "best_params": best_params,
        "metrics": results[best_name]["metrics"],
        "all_results": results,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_pipeline": True,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    if verbose:
        print(f"\n[train] Best model: {best_name} (ROC-AUC={best_auc:.4f})")
        print(f"[train] Saved model -> {MODEL_PATH}")

    return metadata


def _log_plots_to_mlflow(mlflow, pipeline, X_test, y_test, name) -> None:
    """Log confusion matrix and ROC curve as MLflow artifacts."""
    try:
        import tempfile
        from pathlib import Path

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay

        tmp = Path(tempfile.mkdtemp())

        # Confusion matrix
        cm = confusion_matrix(y_test, pipeline.predict(X_test))
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center")
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im)
        cm_path = tmp / f"confusion_matrix_{name}.png"
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
        ax.set_title(f"ROC Curve - {name}")
        roc_path = tmp / f"roc_curve_{name}.png"
        fig.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(roc_path), artifact_path="plots")
    except Exception as exc:  # noqa: BLE001
        print(f"[train] plot logging skipped: {exc}")


if __name__ == "__main__":
    run_training()
