
import os
import random
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from validation import validate_data

SEED = 42
MINIMUM_AUC = 0.60

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets"
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-quality-gate",
        action="store_true"
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    data = pd.read_csv("customers.csv")

    # Step 1: Validate input data
    validate_data(data)
    print("Data validation passed.")

    # Step 2: Separate features and target
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data["churn"].astype(int)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # Step 4: Preprocess numerical and categorical columns
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    # Step 5: Build the complete model pipeline
    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=SEED,
                n_jobs=-1
            )
        )
    ])

    # Step 6: Train
    model.fit(X_train, y_train)

    # Step 7: Evaluate on unseen test data
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print(f"ROC-AUC: {auc:.4f}")

    # Step 8: Enforce the quality gate
    if args.check_quality_gate and auc < MINIMUM_AUC:
        raise SystemExit(
            f"QUALITY GATE FAILED: AUC {auc:.4f} "
            f"is below {MINIMUM_AUC}"
        )

    # Step 9: Save the complete pipeline
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/churn_model.joblib")

    print("Training completed successfully.")
    print("Model saved to artifacts/churn_model.joblib")

if __name__ == "__main__":
    main()
