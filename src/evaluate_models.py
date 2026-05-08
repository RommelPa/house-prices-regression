from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"

TARGET = "SalePrice"
TARGET_LOG = "SalePriceLog"
ID_COLUMN = "Id"


def load_evaluation_inputs() -> tuple[pd.DataFrame, dict, object]:
    """
    Load validation data, feature groups, and trained best model.
    """
    required_files = [
        VALIDATION_MODELING_PATH,
        FEATURE_GROUPS_PATH,
        BEST_MODEL_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing files for evaluation. "
            "Run 'python src/preprocess_data.py' and 'python src/train_models.py' first.\n"
            f"Missing files:\n{missing}"
        )

    validation_data = pd.read_csv(VALIDATION_MODELING_PATH)

    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    best_model = joblib.load(BEST_MODEL_PATH)

    return validation_data, feature_groups, best_model


def build_validation_predictions(
    validation_data: pd.DataFrame,
    feature_groups: dict,
    model: object,
) -> pd.DataFrame:
    """
    Generate validation predictions and error columns.
    """
    target = feature_groups["target"]
    target_log = feature_groups["target_log"]
    id_column = feature_groups["id_column"]

    X_valid = validation_data.drop(columns=[target, target_log, id_column])
    y_valid = validation_data[target]

    y_pred_log = model.predict(X_valid)
    y_pred = np.expm1(y_pred_log)

    predictions = pd.DataFrame(
        {
            "Id": validation_data[id_column],
            "actual_price": y_valid,
            "predicted_price": y_pred,
        }
    )

    predictions["residual"] = (
        predictions["actual_price"] - predictions["predicted_price"]
    )
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["absolute_percentage_error"] = (
        predictions["absolute_error"] / predictions["actual_price"]
    ) * 100

    return predictions


def print_evaluation_summary(predictions: pd.DataFrame) -> None:
    """
    Print validation error summary.
    """
    y_true = predictions["actual_price"]
    y_pred = predictions["predicted_price"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    median_absolute_error = predictions["absolute_error"].median()
    p90_absolute_error = predictions["absolute_error"].quantile(0.90)
    p95_absolute_error = predictions["absolute_error"].quantile(0.95)

    mean_absolute_percentage_error = (
        predictions["absolute_percentage_error"].mean()
    )
    median_absolute_percentage_error = (
        predictions["absolute_percentage_error"].median()
    )

    print("=" * 80)
    print("VALIDATION ERROR ANALYSIS")
    print("=" * 80)

    print("\n1. Main metrics")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}")

    print("\n2. Error distribution")
    print(f"Median absolute error: {median_absolute_error:,.2f}")
    print(f"90th percentile absolute error: {p90_absolute_error:,.2f}")
    print(f"95th percentile absolute error: {p95_absolute_error:,.2f}")
    print(f"Mean absolute percentage error: {mean_absolute_percentage_error:.2f}%")
    print(f"Median absolute percentage error: {median_absolute_percentage_error:.2f}%")

    print("\n3. Largest prediction errors")
    print(
        predictions
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .round(2)
    )


def save_predictions(predictions: pd.DataFrame) -> None:
    """
    Save validation predictions.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    print("\nSaved validation predictions")
    print("-" * 80)
    print(f"Validation predictions: {VALIDATION_PREDICTIONS_PATH}")


def save_diagnostic_figures(predictions: pd.DataFrame) -> None:
    """
    Save diagnostic figures for regression evaluation.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Predicted vs actual
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=predictions,
        x="actual_price",
        y="predicted_price",
        alpha=0.7,
    )

    min_price = min(
        predictions["actual_price"].min(),
        predictions["predicted_price"].min(),
    )
    max_price = max(
        predictions["actual_price"].max(),
        predictions["predicted_price"].max(),
    )

    plt.plot([min_price, max_price], [min_price, max_price], linestyle="--")
    plt.title("Predicted vs Actual Sale Price")
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=300)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions["residual"], bins=40, kde=True)
    plt.axvline(0, linestyle="--")
    plt.title("Residuals Distribution")
    plt.xlabel("Residual: Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_distribution.png", dpi=300)
    plt.close()

    # Residuals vs predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=predictions,
        x="predicted_price",
        y="residual",
        alpha=0.7,
    )
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Predicted Sale Price")
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=300)
    plt.close()

    # Top 20 absolute errors
    top_errors = (
        predictions
        .sort_values("absolute_error", ascending=False)
        .head(20)
        .copy()
    )

    top_errors["Id"] = top_errors["Id"].astype(str)

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=top_errors,
        x="absolute_error",
        y="Id",
    )
    plt.title("Top 20 Validation Predictions by Absolute Error")
    plt.xlabel("Absolute Error")
    plt.ylabel("Property ID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_20_absolute_errors.png", dpi=300)
    plt.close()

    print("\nSaved diagnostic figures")
    print("-" * 80)
    print(f"Predicted vs actual: {FIGURES_DIR / 'predicted_vs_actual.png'}")
    print(f"Residuals distribution: {FIGURES_DIR / 'residuals_distribution.png'}")
    print(f"Residuals vs predicted: {FIGURES_DIR / 'residuals_vs_predicted.png'}")
    print(f"Top 20 absolute errors: {FIGURES_DIR / 'top_20_absolute_errors.png'}")


if __name__ == "__main__":
    validation_data, feature_groups, best_model = load_evaluation_inputs()

    validation_predictions = build_validation_predictions(
        validation_data=validation_data,
        feature_groups=feature_groups,
        model=best_model,
    )

    print_evaluation_summary(validation_predictions)
    save_predictions(validation_predictions)
    save_diagnostic_figures(validation_predictions)