
REQUIRED_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "internet_service",
    "payment_method",
    "churn"
]

def validate_data(data):
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    assert len(data) > 0, "Dataset is empty"
    assert not data[REQUIRED_COLUMNS].isnull().any().any(), \
        "Missing values detected"

    assert set(data["churn"].unique()).issubset({0, 1}), \
        "Churn must contain only 0 or 1"

    assert data["churn"].nunique() == 2, \
        "Both churn classes must be present"

    return True
