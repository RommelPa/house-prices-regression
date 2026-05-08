from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"


def load_train_data(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw training dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Place train.csv inside data/raw/."
        )

    return pd.read_csv(path)


def load_test_data(path: Path = TEST_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw test dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found at {path}. "
            "Place test.csv inside data/raw/."
        )

    return pd.read_csv(path)


def build_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a missing values report for a dataframe.
    """
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    missing_report = (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percent": missing_percent.round(2),
                "dtype": df.dtypes.astype(str),
            }
        )
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )

    return missing_report


def audit_train_data(train_data: pd.DataFrame) -> None:
    """
    Print an audit report for the training dataset.
    """
    print("=" * 80)
    print("HOUSE PRICES TRAIN DATA AUDIT")
    print("=" * 80)

    print("\n1. Dataset shape")
    print(f"Rows: {train_data.shape[0]:,}")
    print(f"Columns: {train_data.shape[1]:,}")

    print("\n2. Duplicate rows")
    print(f"Duplicate rows: {train_data.duplicated().sum():,}")

    print("\n3. Target variable: SalePrice")
    print(train_data["SalePrice"].describe())
    print(f"Missing SalePrice values: {train_data['SalePrice'].isna().sum():,}")

    print("\n4. Feature types")
    feature_data = train_data.drop(columns=["SalePrice"])
    numeric_features = feature_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_data.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Numeric features: {len(numeric_features):,}")
    print(f"Categorical features: {len(categorical_features):,}")

    print("\n5. Missing values report")
    missing_report = build_missing_values_report(train_data)

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report)

    print("\n6. Top numerical correlations with SalePrice")
    numeric_corr = (
        train_data
        .select_dtypes(include=["number"])
        .corr(numeric_only=True)["SalePrice"]
        .drop("SalePrice")
        .sort_values(ascending=False)
    )

    print(numeric_corr.head(15))

    print("\n7. Lowest numerical correlations with SalePrice")
    print(numeric_corr.tail(15))

    print("\n8. Categorical cardinality")
    categorical_cardinality = (
        train_data[categorical_features]
        .nunique(dropna=False)
        .sort_values(ascending=False)
    )

    print(categorical_cardinality)

    print("\n9. Sample rows")
    print(train_data.head())


def audit_test_data(test_data: pd.DataFrame) -> None:
    """
    Print a compact audit report for the test dataset.
    """
    print("\n" + "=" * 80)
    print("HOUSE PRICES TEST DATA AUDIT")
    print("=" * 80)

    print("\n1. Dataset shape")
    print(f"Rows: {test_data.shape[0]:,}")
    print(f"Columns: {test_data.shape[1]:,}")

    print("\n2. Missing values report")
    missing_report = build_missing_values_report(test_data)

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report)

    print("\n3. Duplicate rows")
    print(f"Duplicate rows: {test_data.duplicated().sum():,}")


if __name__ == "__main__":
    train = load_train_data()
    test = load_test_data()

    audit_train_data(train)
    audit_test_data(test)