
from pathlib import Path

import numpy as np
import pandas as pd

# Always save the CSV beside this Python file.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "customers.csv"


def generate_data():
    rng = np.random.default_rng(42)
    rows = 2000

    # Generate numerical features.
    tenure = rng.integers(1, 73, rows)
    charges = rng.uniform(25, 120, rows).round(2)
    calls = rng.poisson(2, rows)

    # Generate categorical features.
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        rows,
        p=[0.55, 0.25, 0.20],
    )

    internet = rng.choice(
        ["Fiber", "DSL", "None"],
        rows,
    )

    # Calculate churn probability.
    score = (
        -1.0
        - 0.045 * tenure
        + 0.035 * calls
        + 0.018 * (charges - 70)
        + 1.1 * (contracts == "Month-to-month")
        - 0.8 * (contracts == "Two year")
        + 0.5 * (internet == "Fiber")
    )

    probabilities = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probabilities)

    # Create the dataset.
    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": charges,
        "support_calls": calls,
        "contract": contracts,
        "internet_service": internet,
        "churn": churn,
    })

    # Save the dataset.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    # Read back the exact CSV file.
    saved_df = pd.read_csv(CSV_PATH, keep_default_na=False)

    # Display diagnostic information.
    print("\nGenerator file:", Path(__file__).resolve())
    print("CSV file:", CSV_PATH)
    print("Dataset shape:", saved_df.shape)
    print("Total missing values:", int(saved_df.isna().sum().sum()))

    print("\nMissing values by column:")
    print(saved_df.isna().sum())

    print("\nRows containing missing values:")
    missing_rows = saved_df[saved_df.isna().any(axis=1)]
    print(missing_rows.head(10).to_string())

    print("\nFirst 10 raw CSV lines:")
    print(CSV_PATH.read_text(encoding="utf-8").splitlines()[:10])

      # Validate the generated dataset.
    assert saved_df.shape == (2000, 6), (
        f"Unexpected dataset shape: {saved_df.shape}"
    )

    # Diagnostic mode: display missing values without stopping.
    print("\nMissing values by column:")
    print(saved_df.isna().sum())

    print("\nRows containing missing values:")
    missing_rows = saved_df[saved_df.isna().any(axis=1)]
    print(missing_rows.head(10).to_string())

    print("\nFirst 10 raw CSV lines:")
    print(CSV_PATH.read_text(encoding="utf-8").splitlines()[:10])

    print("\nTotal missing values:", int(saved_df.isna().sum().sum()))


if __name__ == "__main__":
    generate_data()