from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from train_models import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_MODELING_PATH = PROCESSED_DATA_DIR / "train_modeling.csv"
VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

CROSS_VALIDATION_METRICS_PATH = REPORTS_DIR / "cross_validation_metrics.csv"
CROSS_VALIDATION_SUMMARY_PATH = REPORTS_DIR / "cross_validation_summary.csv"

N_SPLITS = 5
RANDOM_STATE = 42


def load_full_training_data() -> tuple[pd.DataFrame, dict]:
    """
    Load train and validation splits and combine them for cross-validation.

    This uses only the original Kaggle training data. The Kaggle test dataset
    remains untouched because it does not contain SalePrice.
    """
    required_files = [
        TRAIN_MODELING_PATH,
        VALIDATION_MODELING_PATH,
        FEATURE_GROUPS_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing processed modeling files. "
            "Run 'python src/preprocess_data.py' first.\n"
            f"Missing files:\n{missing}"
        )

    train_data = pd.read_csv(TRAIN_MODELING_PATH)
    validation_data = pd.read_csv(VALIDATION_MODELING_PATH)

    full_training_data = pd.concat(
        [train_data, validation_data],
        axis=0,
        ignore_index=True,
    )

    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    return full_training_data, feature_groups


def split_features_target(
    data: pd.DataFrame,
    feature_groups: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target columns.
    """
    target = feature_groups["target"]
    target_log = feature_groups["target_log"]
    id_column = feature_groups["id_column"]

    X = data.drop(columns=[target, target_log, id_column])
    y = data[target]
    y_log = data[target_log]

    return X, y, y_log


def evaluate_fold_predictions(
    model_name: str,
    fold: int,
    y_true: pd.Series,
    y_pred_log: np.ndarray,
) -> dict:
    """
    Evaluate fold predictions on original price scale.
    """
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "fold": fold,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    y_log: pd.Series,
    feature_groups: dict,
) -> pd.DataFrame:
    """
    Run K-Fold cross-validation for all model pipelines.
    """
    models = build_models(feature_groups)

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model_pipeline in models.items():
        print(f"Cross-validating {model_name}...")

        for fold_index, (train_index, valid_index) in enumerate(kfold.split(X), start=1):
            X_train_fold = X.iloc[train_index]
            X_valid_fold = X.iloc[valid_index]

            y_train_log_fold = y_log.iloc[train_index]
            y_valid_fold = y.iloc[valid_index]

            fold_model = clone(model_pipeline)
            fold_model.fit(X_train_fold, y_train_log_fold)

            y_pred_log = fold_model.predict(X_valid_fold)

            fold_metrics = evaluate_fold_predictions(
                model_name=model_name,
                fold=fold_index,
                y_true=y_valid_fold,
                y_pred_log=y_pred_log,
            )

            results.append(fold_metrics)

    return pd.DataFrame(results)


def build_cross_validation_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation metrics by model.
    """
    summary = (
        cv_results
        .groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .sort_values("rmse_mean", ascending=True)
        .reset_index(drop=True)
    )

    return summary


def save_cross_validation_outputs(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> None:
    """
    Save cross-validation metrics, summary, and figure.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cv_results.to_csv(CROSS_VALIDATION_METRICS_PATH, index=False)
    cv_summary.to_csv(CROSS_VALIDATION_SUMMARY_PATH, index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_summary, x="rmse_mean", y="model")
    plt.title("Cross-Validation RMSE by Model")
    plt.xlabel("Mean RMSE")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_validation_rmse_by_model.png", dpi=300)
    plt.close()

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Fold metrics: {CROSS_VALIDATION_METRICS_PATH}")
    print(f"Summary metrics: {CROSS_VALIDATION_SUMMARY_PATH}")
    print(f"Figure: {FIGURES_DIR / 'cross_validation_rmse_by_model.png'}")


def print_cross_validation_summary(cv_summary: pd.DataFrame) -> None:
    """
    Print cross-validation summary.
    """
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    formatted_summary = cv_summary.copy()

    metric_columns = [
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "r2_mean",
        "r2_std",
    ]

    for column in metric_columns:
        formatted_summary[column] = formatted_summary[column].round(4)

    print(formatted_summary)


if __name__ == "__main__":
    full_data, feature_groups = load_full_training_data()
    X, y, y_log = split_features_target(full_data, feature_groups)

    cv_results = run_cross_validation(
        X=X,
        y=y,
        y_log=y_log,
        feature_groups=feature_groups,
    )

    cv_summary = build_cross_validation_summary(cv_results)

    print_cross_validation_summary(cv_summary)
    save_cross_validation_outputs(cv_results, cv_summary)