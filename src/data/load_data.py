from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset location
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Parameters
    ----------
    path : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded customer churn dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the dataset does not contain the expected schema.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    # Validate expected schema
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_columns)}"
        )

    # Check for unexpected columns
    unexpected_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if unexpected_columns:
        print(
            f"Warning: unexpected columns detected: "
            f"{sorted(unexpected_columns)}"
        )

    # Basic dataset information
    print("=" * 70)
    print("CUSTOMER CHURN DATASET - INGESTION REPORT")
    print("=" * 70)

    print(f"Dataset path       : {path}")
    print(f"Number of rows     : {df.shape[0]:,}")
    print(f"Number of columns  : {df.shape[1]:,}")

    print("\nDuplicate customer IDs:")
    print(df["customerID"].duplicated().sum())

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nData types:")
    print(df.dtypes)

    print("=" * 70)

    return df


if __name__ == "__main__":
    df = load_raw_data()