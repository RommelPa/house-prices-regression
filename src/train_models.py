from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_MODELING_PATH = PROCESSED_DATA_DIR / "train_modeling.csv"
VALIDATION_MODELING_PATH = PROCESSED_DATA_DIR / "validation_modeling.csv"
FEATURE_GROUPS_PATH = PROCESSED_DATA_DIR / "feature_groups.json"

MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FINAL_MODEL_SELECTION_PATH = REPORTS_DIR / "final_model_selection.csv"

FINAL_MODEL_NAME = "random_forest"

RANDOM_STATE = 42


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load train/validation modeling datasets and feature groups.
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

    with open(FEATURE_GROUPS_PATH, "r", encoding="utf-8") as file:
        feature_groups = json.load(file)

    return train_data, validation_data, feature_groups


def split_features_target(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_groups: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and targets for train and validation datasets.
    """
    target = feature_groups["target"]
    target_log = feature_groups["target_log"]
    id_column = feature_groups["id_column"]

    drop_columns = [target, target_log, id_column]

    X_train = train_data.drop(columns=drop_columns)
    y_train = train_data[target]
    y_train_log = train_data[target_log]

    X_valid = validation_data.drop(columns=drop_columns)
    y_valid = validation_data[target]
    y_valid_log = validation_data[target_log]

    return X_train, y_train, y_train_log, X_valid, y_valid, y_valid_log


def build_preprocessor(feature_groups: dict, scale_numeric: bool) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric and categorical preprocessing.
    """
    categorical_none_features = feature_groups["categorical_none_features"]
    categorical_most_frequent_features = feature_groups[
        "categorical_most_frequent_features"
    ]
    numeric_zero_features = feature_groups["numeric_zero_features"]
    numeric_median_features = feature_groups["numeric_median_features"]

    if scale_numeric:
        numeric_zero_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )

        numeric_median_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_zero_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ]
        )

        numeric_median_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_none_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    categorical_most_frequent_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_zero", numeric_zero_pipeline, numeric_zero_features),
            ("numeric_median", numeric_median_pipeline, numeric_median_features),
            (
                "categorical_none",
                categorical_none_pipeline,
                categorical_none_features,
            ),
            (
                "categorical_most_frequent",
                categorical_most_frequent_pipeline,
                categorical_most_frequent_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_models(feature_groups: dict) -> dict[str, Pipeline]:
    """
    Build model pipelines.
    """
    linear_preprocessor = build_preprocessor(feature_groups, scale_numeric=True)
    tree_preprocessor = build_preprocessor(feature_groups, scale_numeric=False)

    models = {
        "baseline_median": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "lasso_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred_log: np.ndarray,
) -> dict:
    """
    Evaluate log-scale predictions after converting them back to price scale.
    """
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_and_evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """
    Train all models and evaluate them on validation data.
    """
    results = []
    trained_models = {}

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")

        pipeline.fit(X_train, y_train_log)
        y_pred_log = pipeline.predict(X_valid)

        metrics = evaluate_predictions(
            model_name=model_name,
            y_true=y_valid,
            y_pred_log=y_pred_log,
        )

        results.append(metrics)
        trained_models[model_name] = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("rmse", ascending=True)
        .reset_index(drop=True)
    )

    return results_df, trained_models


def save_outputs(
    results_df: pd.DataFrame,
    trained_models: dict[str, Pipeline],
) -> None:
    """
    Save model metrics and the final selected model.

    The final model is selected based on cross-validation stability, not only
    on a single validation split.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(MODEL_METRICS_PATH, index=False)

    validation_best_model_name = results_df.loc[0, "model"]

    if FINAL_MODEL_NAME not in trained_models:
        raise ValueError(
            f"Final model '{FINAL_MODEL_NAME}' was not found in trained models."
        )

    final_model = trained_models[FINAL_MODEL_NAME]
    joblib.dump(final_model, BEST_MODEL_PATH)

    final_model_selection = pd.DataFrame(
        [
            {
                "validation_best_model": validation_best_model_name,
                "final_selected_model": FINAL_MODEL_NAME,
                "selection_reason": (
                    "Random Forest was selected as the final predictive model "
                    "because cross-validation showed better RMSE stability than "
                    "linear models, even though Linear Regression performed best "
                    "on a single validation split."
                ),
            }
        ]
    )

    final_model_selection.to_csv(FINAL_MODEL_SELECTION_PATH, index=False)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Model metrics: {MODEL_METRICS_PATH}")
    print(f"Final model selection: {FINAL_MODEL_SELECTION_PATH}")
    print(f"Final model: {BEST_MODEL_PATH}")
    print(f"Validation best model: {validation_best_model_name}")
    print(f"Final selected model: {FINAL_MODEL_NAME}")


def print_results(results_df: pd.DataFrame) -> None:
    """
    Print model comparison results.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    formatted_results = results_df.copy()
    formatted_results["mae"] = formatted_results["mae"].round(2)
    formatted_results["rmse"] = formatted_results["rmse"].round(2)
    formatted_results["r2"] = formatted_results["r2"].round(4)

    print(formatted_results)


if __name__ == "__main__":
    train_data, validation_data, feature_groups = load_modeling_data()

    (
        X_train,
        _,
        y_train_log,
        X_valid,
        y_valid,
        _,
    ) = split_features_target(
        train_data=train_data,
        validation_data=validation_data,
        feature_groups=feature_groups,
    )

    model_pipelines = build_models(feature_groups)

    model_results, fitted_models = train_and_evaluate_models(
        models=model_pipelines,
        X_train=X_train,
        y_train_log=y_train_log,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    print_results(model_results)
    save_outputs(model_results, fitted_models)