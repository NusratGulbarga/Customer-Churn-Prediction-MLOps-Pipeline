
import os
import random
import argparse
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from validation import validate_data

SEED = 42
MINIMUM_AUC = 0.60

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_length",
    "late_payments",
    "usage_score",
    "discount_rate",
    "service_count",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-quality-gate", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    # Keep the experiment and run history in the current workspace.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Customer-Churn-MLOps")

    data = pd.read_csv("customers.csv")
    validate_data(data)

    X = data[FEATURES]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="random-forest-churn") as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 8,
            "random_seed": SEED,
            "dataset_rows": len(data),
        })

        mlflow.log_metrics(metrics)
        mlflow.set_tag("dataset", "synthetic-v1")
        mlflow.set_tag("quality_gate", "enabled")

        os.makedirs("artifacts", exist_ok=True)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        joblib.dump(model, "artifacts/churn_model.joblib")

        print("MLflow run ID:", run.info.run_id)
        print("Metrics:", metrics)

        if args.check_quality_gate and metrics["auc"] < MINIMUM_AUC:
            raise SystemExit(
                f"QUALITY GATE FAILED: AUC={metrics['auc']:.4f}"
            )

        print("QUALITY GATE PASSED")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
