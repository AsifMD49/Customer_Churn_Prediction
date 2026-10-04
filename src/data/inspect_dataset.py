from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def main() -> None:
    print("=" * 70)
    print("CUSTOMER CHURN DATASET INSPECTION")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n\n"
            "Please place the CSV inside data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset path : {DATA_PATH}")
    print(f"Rows         : {df.shape[0]:,}")
    print(f"Columns      : {df.shape[1]:,}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "Churn" in df.columns:
        print("\nTarget distribution:")
        print(df["Churn"].value_counts())

        print("\nTarget distribution (%):")
        print(
            df["Churn"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )


if __name__ == "__main__":
    main()