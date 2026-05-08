from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"
DATA_DESCRIPTION_PATH = RAW_DATA_DIR / "data_description.txt"


def load_train_data(path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw training dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Download train.csv from Kaggle and place it in data/raw/."
        )

    return pd.read_csv(path)


def load_test_data(path: Path = TEST_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw test dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Test data not found at {path}. "
            "Download test.csv from Kaggle and place it in data/raw/."
        )

    return pd.read_csv(path)


def validate_required_files() -> None:
    """
    Validate that all expected raw files exist.
    """
    required_files = [
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        DATA_DESCRIPTION_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing raw dataset files:\n"
            f"{missing}\n\n"
            "Expected files: train.csv, test.csv, data_description.txt"
        )


if __name__ == "__main__":
    validate_required_files()

    train_data = load_train_data()
    test_data = load_test_data()

    print("Raw dataset files found successfully.")
    print(f"Train shape: {train_data.shape[0]:,} rows, {train_data.shape[1]:,} columns")
    print(f"Test shape: {test_data.shape[0]:,} rows, {test_data.shape[1]:,} columns")
    print("\nTrain columns:")
    print(list(train_data.columns))