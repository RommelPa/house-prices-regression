from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

TRAIN_MODELING_PATH = PROCESSED_DATA_DIR / "train_modeling.csv"
VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
TEST_MODELING_PATH = PROCESSED_DATA_DIR / "test_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

TARGET = "SalePrice"
TARGET_LOG = "SalePriceLog"
ID_COLUMN = "Id"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


CATEGORICAL_NONE_FEATURES = [
    "Alley",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

NUMERIC_ZERO_FEATURES = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw train and test datasets.
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_PATH}. "
            "Place train.csv inside data/raw/."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found at {TEST_DATA_PATH}. "
            "Place test.csv inside data/raw/."
        )

    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    return train_data, test_data


def validate_input_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Validate basic input requirements.
    """
    if TARGET not in train_data.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")

    if TARGET in test_data.columns:
        raise ValueError(f"Target column '{TARGET}' should not exist in test data.")

    if ID_COLUMN not in train_data.columns or ID_COLUMN not in test_data.columns:
        raise ValueError(f"ID column '{ID_COLUMN}' must exist in train and test data.")

    if train_data[TARGET].isna().any():
        raise ValueError("Training data contains missing target values.")


def add_target_log(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed target to reduce skewness.
    """
    train_data = train_data.copy()
    train_data[TARGET_LOG] = np.log1p(train_data[TARGET])

    return train_data


def build_feature_groups(train_data: pd.DataFrame) -> dict:
    """
    Build feature groups for downstream scikit-learn preprocessing pipelines.
    """
    feature_data = train_data.drop(columns=[TARGET, TARGET_LOG])

    numeric_features = feature_data.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_data.select_dtypes(exclude=["number"]).columns.tolist()

    # The ID column is not a predictive feature.
    numeric_features = [col for col in numeric_features if col != ID_COLUMN]
    categorical_features = [col for col in categorical_features if col != ID_COLUMN]

    categorical_none_features = [
        col for col in CATEGORICAL_NONE_FEATURES if col in categorical_features
    ]

    numeric_zero_features = [
        col for col in NUMERIC_ZERO_FEATURES if col in numeric_features
    ]

    categorical_most_frequent_features = [
        col for col in categorical_features if col not in categorical_none_features
    ]

    numeric_median_features = [
        col for col in numeric_features if col not in numeric_zero_features
    ]

    feature_groups = {
        "id_column": ID_COLUMN,
        "target": TARGET,
        "target_log": TARGET_LOG,
        "categorical_none_features": categorical_none_features,
        "categorical_most_frequent_features": categorical_most_frequent_features,
        "numeric_zero_features": numeric_zero_features,
        "numeric_median_features": numeric_median_features,
    }

    return feature_groups


def create_modeling_splits(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train and validation sets.
    """
    train_split, validation_split = train_test_split(
        train_data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)


def save_processed_outputs(
    train_split: pd.DataFrame,
    validation_split: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_groups: dict,
) -> None:
    """
    Save modeling splits and feature groups.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_split.to_csv(TRAIN_MODELING_PATH, index=False)
    validation_split.to_csv(VALIDATION_MODELING_PATH, index=False)
    test_data.to_csv(TEST_MODELING_PATH, index=False)

    with open(FEATURE_GROUPS_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_groups, file, indent=4)


def print_preprocessing_summary(
    train_data: pd.DataFrame,
    train_split: pd.DataFrame,
    validation_split: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_groups: dict,
) -> None:
    """
    Print preprocessing summary.
    """
    print("=" * 80)
    print("HOUSE PRICES PREPROCESSING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset shapes")
    print(f"Original train: {train_data.shape[0]:,} rows, {train_data.shape[1]:,} columns")
    print(f"Train split: {train_split.shape[0]:,} rows, {train_split.shape[1]:,} columns")
    print(
        f"Validation split: {validation_split.shape[0]:,} rows, "
        f"{validation_split.shape[1]:,} columns"
    )
    print(f"Test data: {test_data.shape[0]:,} rows, {test_data.shape[1]:,} columns")

    print("\n2. Target distribution")
    print(train_data[TARGET].describe())
    print("\nLog target distribution")
    print(train_data[TARGET_LOG].describe())

    print("\n3. Feature groups")
    for group_name, columns in feature_groups.items():
        if isinstance(columns, list):
            print(f"{group_name}: {len(columns)}")
        else:
            print(f"{group_name}: {columns}")

    print("\n4. Files saved")
    print(f"Train modeling data: {TRAIN_MODELING_PATH}")
    print(f"Validation modeling data: {VALIDATION_MODELING_PATH}")
    print(f"Test modeling data: {TEST_MODELING_PATH}")
    print(f"Feature groups: {FEATURE_GROUPS_PATH}")


if __name__ == "__main__":
    train, test = load_raw_data()
    validate_input_data(train, test)

    train = add_target_log(train)
    feature_groups = build_feature_groups(train)

    train_split, validation_split = create_modeling_splits(train)

    save_processed_outputs(
        train_split=train_split,
        validation_split=validation_split,
        test_data=test,
        feature_groups=feature_groups,
    )

    print_preprocessing_summary(
        train_data=train,
        train_split=train_split,
        validation_split=validation_split,
        test_data=test,
        feature_groups=feature_groups,
    )