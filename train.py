
import argparse
import json
import random
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

SEED = 42
MINIMUM_AUC = 0.60
EXPERIMENT_NAME = "Customer Churn Prediction"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
MLFLOW_DIR = BASE_DIR / "mlruns"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Use a single, consistent MLflow tracking directory.
mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

NUMERIC = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL = [
    "contract",
    "internet_service",
]

FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


# ============================================================
# 3. DATA VALIDATION
# ============================================================

def validate_data(df):
    """Validate the input dataset before training."""

    required_columns = FEATURES + [TARGET]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[required_columns].eq("").any().any():
        raise ValueError(
            "Dataset contains empty strings."
        )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Convert target to numeric and validate it.
    try:
        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Target must contain numeric 0 and 1 values."
        ) from exc

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Target must contain both classes."
        )

    # Validate numeric features.
    try:
        numeric_values = df[NUMERIC].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Numeric features contain invalid values."
        ) from exc

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Numeric features must contain finite values."
        )

    print("\nDATA VALIDATION PASSED")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(FEATURES)}")
    print("Missing values: 0")

    return True


# ============================================================
# 4. BUILD THE MACHINE LEARNING PIPELINE
# ============================================================

def build_model():
    """Build preprocessing and Logistic Regression."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    return model


# ============================================================
# 5. TRAIN, EVALUATE, AND LOG WITH MLFLOW
# ============================================================

def main(check_quality_gate=False):
    """Execute the complete training pipeline."""

    random.seed(SEED)
    np.random.seed(SEED)

    print("\n" + "=" * 55)
    print("CUSTOMER CHURN PREDICTION - MLOPS PIPELINE")
    print("=" * 55)

    print(f"Dataset: {DATA_PATH}")
    print(f"MLflow tracking: {MLFLOW_DIR}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    # Load dataset. Preserve the category "None".
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python generate_data.py first."
        )

    df = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
    )

    validate_data(df)

    X = df[FEATURES]
    y = df[TARGET]

    # Split into training and testing datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    model = build_model()

    # Start an MLflow experiment run.
    with mlflow.start_run() as run:

        # Log model parameters.
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "max_iter": 1000,
                "test_size": 0.20,
                "random_state": SEED,
                "minimum_auc": MINIMUM_AUC,
            }
        )

        # Train the model.
        print("\nTraining model...")
        model.fit(X_train, y_train)

        # Make predictions.
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics.
        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "minimum_auc": MINIMUM_AUC,
        }

        # Log metrics to MLflow.
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
            }
        )

        # Create output directory and save model.
        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        METRICS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(model, MODEL_PATH)

        with METRICS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=2)

        # Log the model using the MLflow 2.x-compatible argument.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        # Log the metrics JSON as an artifact.
        mlflow.log_artifact(str(METRICS_PATH))

        # Display results.
        print("\n" + "-" * 55)
        print("TRAINING COMPLETED")
        print("-" * 55)

        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")

        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))

        print(f"\nModel saved: {MODEL_PATH}")
        print(f"Metrics saved: {METRICS_PATH}")

    # Enforce the quality gate when requested.
    if check_quality_gate:
        if metrics["auc"] < MINIMUM_AUC:
            raise SystemExit(
                f"QUALITY GATE FAILED: AUC "
                f"{metrics['auc']:.3f} is below "
                f"{MINIMUM_AUC:.2f}"
            )

        print(
            f"\nQUALITY GATE PASSED: AUC "
            f"{metrics['auc']:.3f} >= {MINIMUM_AUC:.2f}"
        )
    else:
        print(
            f"\nTraining completed. AUC: "
            f"{metrics['auc']:.3f}"
        )

    print("\nMLflow UI: http://127.0.0.1:5000")
    print("Pipeline execution finished.")

    return metrics


# ============================================================
# 6. COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction MLOps Pipeline"
    )

    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail if ROC-AUC is below 0.60.",
    )

    args = parser.parse_args()

    main(
        check_quality_gate=args.check_quality_gate
    )
