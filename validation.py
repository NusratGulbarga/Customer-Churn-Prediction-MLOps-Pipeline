
import pandas as pd

REQUIRED_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_length",
    "late_payments",
    "usage_score",
    "discount_rate",
    "service_count",
    "churn"
]

def validate_data(data):
    assert set(REQUIRED_COLUMNS).issubset(data.columns)
    assert len(data) > 0
    assert not data[REQUIRED_COLUMNS].isnull().any().any()
    assert set(data["churn"].unique()).issubset({0, 1})
    assert data["churn"].nunique() == 2

    return True
