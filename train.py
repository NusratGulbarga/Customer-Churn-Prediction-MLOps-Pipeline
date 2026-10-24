
import os
import random
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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
    "service_count"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-quality-gate", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    data = pd.read_csv("customers.csv")
    validate_data(data)

    X = data[FEATURES]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print(f"ROC-AUC: {auc:.4f}")

    if args.check_quality_gate and auc < MINIMUM_AUC:
        raise SystemExit(
            f"QUALITY GATE FAILED: {auc:.4f} < {MINIMUM_AUC}"
        )

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/churn_model.joblib")

    print("Training completed.")
    print("Model saved to artifacts/churn_model.joblib")

if __name__ == "__main__":
    main()
