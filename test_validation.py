
import pandas as pd
import pytest

from validation import validate_data, REQUIRED_COLUMNS

def valid_dataset():
    return pd.DataFrame({
        column: [0, 1, 0, 1]
        for column in REQUIRED_COLUMNS
    })

def test_valid_data_passes():
    assert validate_data(valid_dataset()) is True

def test_missing_column_fails():
    data = valid_dataset().drop(columns=["churn"])

    with pytest.raises(AssertionError):
        validate_data(data)

def test_missing_values_fail():
    data = valid_dataset()
    data.loc[0, "monthly_charges"] = float("nan")

    with pytest.raises(AssertionError):
        validate_data(data)

def test_invalid_target_fails():
    data = valid_dataset()
    data["churn"] = [0, 1, 2, 1]

    with pytest.raises(AssertionError):
        validate_data(data)
