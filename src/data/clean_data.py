from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.data.load_data import load_raw_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_customer_churn_cleaned.csv"
)


NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
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
]

EXPECTED_COLUMNS = [
    "customerID",
    *CATEGORICAL_COLUMNS[:3],
    "SeniorCitizen",
    "tenure",
    *CATEGORICAL_COLUMNS[3:],
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the Telco Customer Churn dataset.

    The input DataFrame is not modified in place.

    Cleaning decisions:
    - Remove surrounding whitespace from string values.
    - Convert blank strings to missing values.
    - Convert numerical columns to numeric types.
    - Preserve legitimate categorical values such as
      'No internet service' and 'No phone service'.
    - Remove exact duplicate rows.
    - Reject duplicate customer IDs.
    - Validate the target and key business constraints.
    """

    df = df.copy()

    print("\n" + "=" * 70)
    print("DATA CLEANING AND VALIDATION")
    print("=" * 70)

    original_rows = len(df)

    # -------------------------------------------------
    # 1. Validate the dataset schema
    # -------------------------------------------------

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    if len(df.columns) != len(set(df.columns)):
        raise ValueError("Duplicate column names detected.")

    # -------------------------------------------------
    # 2. Normalize string values
    # -------------------------------------------------

    for column in df.select_dtypes(include=["object", "str"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        df[column] = df[column].replace("", np.nan)

    # -------------------------------------------------
    # 3. Convert numerical columns
    # -------------------------------------------------

    for column in NUMERIC_COLUMNS:
        original_nonmissing = df[column].notna()

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # Do not silently turn invalid nonblank values into NaN.
        invalid_values = original_nonmissing & converted.isna()

        if invalid_values.any():
            examples = df.loc[invalid_values, column].head(5).tolist()

            raise ValueError(
                f"Invalid numerical values found in {column}: {examples}"
            )

        df[column] = converted

    # -------------------------------------------------
    # 4. Remove exact duplicate rows
    # -------------------------------------------------

    exact_duplicates = int(df.duplicated().sum())

    if exact_duplicates:
        df = df.drop_duplicates().copy()

    # -------------------------------------------------
    # 5. Validate customer identity
    # -------------------------------------------------

    if df["customerID"].isna().any():
        raise ValueError("Customer IDs cannot be missing.")

    if df["customerID"].duplicated().any():
        duplicated_ids = (
            df.loc[
                df["customerID"].duplicated(keep=False),
                "customerID",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate customer IDs detected. "
            f"Examples: {duplicated_ids[:10]}"
        )

    # -------------------------------------------------
    # 6. Validate the target
    # -------------------------------------------------

    valid_target_values = {"Yes", "No"}

    if df["Churn"].isna().any():
        raise ValueError("The Churn target contains missing values.")

    invalid_targets = set(df["Churn"].unique()) - valid_target_values

    if invalid_targets:
        raise ValueError(
            f"Unexpected Churn values: {sorted(invalid_targets)}"
        )

    # -------------------------------------------------
    # 7. Validate business constraints
    # -------------------------------------------------

    if not df["tenure"].between(0, 1000).all():
        raise ValueError("Unexpected tenure values detected.")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("MonthlyCharges cannot be negative.")

    if (df["TotalCharges"].dropna() < 0).any():
        raise ValueError("TotalCharges cannot be negative.")

    # In this dataset, zero-tenure customers have blank
    # TotalCharges. Verify that relationship before imputing.
    zero_tenure = df["tenure"].eq(0)

    nonzero_tenure_missing_charges = (
        df["TotalCharges"].isna() & ~zero_tenure
    )

    if nonzero_tenure_missing_charges.any():
        raise ValueError(
            "Missing TotalCharges found for customers "
            "whose tenure is greater than zero."
        )

    # The raw dataset has no TotalCharges for zero-tenure
    # customers. Here we represent the absence of accumulated
    # charges as 0.0, based on this dataset's business context.
    zero_tenure_missing_charges = (
        zero_tenure & df["TotalCharges"].isna()
    )

    imputed_count = int(zero_tenure_missing_charges.sum())

    df.loc[
        zero_tenure_missing_charges,
        "TotalCharges",
    ] = 0.0

    # -------------------------------------------------
    # 8. Validate the final DataFrame
    # -------------------------------------------------

    schema = pa.DataFrameSchema(
        {
            "customerID": pa.Column(str, nullable=False),
            "tenure": pa.Column(float, nullable=False, coerce=True),
            "MonthlyCharges": pa.Column(float, nullable=False),
            "TotalCharges": pa.Column(float, nullable=False),
            "Churn": pa.Column(
                str,
                checks=pa.Check.isin(["Yes", "No"]),
                nullable=False,
            ),
        },
        strict=False,
        coerce=True,
    )

    df = schema.validate(df, lazy=True)

    # -------------------------------------------------
    # 9. Print the cleaning report
    # -------------------------------------------------

    print(f"Original rows: {original_rows:,}")
    print(f"Final rows: {len(df):,}")
    print(f"Exact duplicate rows removed: {exact_duplicates:,}")
    print(f"Zero-tenure TotalCharges values set to zero: {imputed_count:,}")

    print("\nMissing values after cleaning:")

    remaining_missing = df.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    if remaining_missing.empty:
        print("No missing values.")
    else:
        print(remaining_missing)

    print("\nFinal data types:")
    print(df.dtypes)

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())

    print("=" * 70)

    return df


def save_cleaned_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the cleaned dataset to the processed-data directory."""

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f"\nCleaned dataset saved to: {path}")


if __name__ == "__main__":
    raw_df = load_raw_data()

    cleaned_df = clean_data(raw_df)

    save_cleaned_data(cleaned_df)