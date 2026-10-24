

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train import build_model, validate_data


@pytest.fixture
def sample_data():
    """Create a small valid customer churn dataset."""
    return pd.DataFrame(
        {
            "tenure": [2, 12, 24, 36, 5, 48, 10, 30],
            "monthly_charges": [
                80.0, 60.0, 70.0, 45.0,
                90.0, 55.0, 85.0, 50.0,
            ],
            "support_calls": [5, 2, 1, 0, 6, 1, 4, 0],
            "contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "internet_service": [
                "Fiber",
                "DSL",
                "None",
                "Fiber",
                "DSL",
                "None",
                "Fiber",
                "DSL",
            ],
            "churn": [1, 0, 0, 0, 1, 0, 1, 0],
        }
    )


def test_valid_data_passes_validation(sample_data):
    assert validate_data(sample_data.copy()) is True


def test_missing_column_is_rejected(sample_data):
    invalid_data = sample_data.drop(columns=["tenure"])

    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(invalid_data)


def test_missing_values_are_rejected(sample_data):
    invalid_data = sample_data.copy()
    invalid_data.loc[0, "contract"] = np.nan

    with pytest.raises(ValueError, match="contains missing values"):
        validate_data(invalid_data)


def test_target_must_have_both_classes(sample_data):
    invalid_data = sample_data.copy()
    invalid_data["churn"] = 0

    with pytest.raises(ValueError, match="both classes"):
        validate_data(invalid_data)


def test_model_can_train_and_predict(sample_data):
    X = sample_data[
        [
            "tenure",
            "monthly_charges",
            "support_calls",
            "contract",
            "internet_service",
        ]
    ]
    y = sample_data["churn"]

    model = build_model()
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(sample_data)
    assert probabilities.shape == (len(sample_data), 2)
    assert set(predictions).issubset({0, 1})


def test_dataset_generator_exists():
    project_dir = Path(__file__).resolve().parents[1]

    assert (project_dir / "generate_data.py").exists()
