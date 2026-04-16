# Change python behavior
from __future__ import annotations

# Standard library imports
from pathlib import Path
from typing import Dict, Tuple

# General library imports
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Default dataset file
DEFAULT_RAW = RAW_DIR / "cardio_train.csv"


# Ensures the directories exist
def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# Returns raw dataset file, if it exists
def locate_raw_dataset() -> Path:
    if DEFAULT_RAW.exists():
        return DEFAULT_RAW
    raise FileNotFoundError(f"Could not find cardio dataset at {DEFAULT_RAW}")


# Load data from raw dataset file
def load_raw_cardio() -> pd.DataFrame:
    return pd.read_csv(locate_raw_dataset(), sep=";")


# Load train and test data
def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    return x_train, x_test, y_train, y_test


# Perform data quality audit
def data_audit(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_rows_without_id": int(df.drop(columns=["id"]).duplicated().sum()),
        "target_0": int((df["cardio"] == 0).sum()),
        "target_1": int((df["cardio"] == 1).sum()),
        "bad_bp": int(
            ((df["ap_hi"] < df["ap_lo"]) | (df["ap_hi"] <= 0) | (df["ap_lo"] <= 0)).sum()
        ),
    }


# Clean cardio dataset, keeps realistic data
def clean_cardio(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["ap_hi"] >= df["ap_lo"])
        & (df["ap_hi"] > 0)
        & (df["ap_lo"] > 0)
        & (df["ap_hi"] <= 250)
        & (df["ap_lo"] <= 200)
        & (df["height"].between(120, 220))
        & (df["weight"].between(30, 250))
    )
    return df.loc[mask].copy()


# Splits features and target and create new features
def feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["cardio"]).copy()
    y = df["cardio"].astype(int).copy()
    if "id" in x.columns:
        x = x.drop(columns=["id"])
    x["age_years"] = x["age"] / 365.25
    x["bmi"] = x["weight"] / ((x["height"] / 100.0) ** 2)
    return x, y


# Build preprocessing pipeline for machine learning models
def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


# Create a smaller dataset while keeping class balance
def stratified_subsample(
    x: np.ndarray, y: np.ndarray, max_points: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_points:
        return x, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_points, random_state=random_state)
    idx, _ = next(sss.split(x, y))
    return x[idx], y[idx]


# Saves metrics as CSV files
def save_metrics_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    path = METRICS_DIR / filename
    df.to_csv(path, index=False)
    return path


# Saves model as joblib files
def save_model(model: BaseEstimator, modelname: str) -> Path:
    ensure_dirs()
    path = MODELS_DIR / modelname
    joblib.dump(model, path)
    return path


# Loads model from joblib files
def load_model(filename: str) -> BaseEstimator:
    path = MODELS_DIR / filename
    return joblib.load(path)


# Load trained models from previous steps per round
def load_models_by_round() -> Tuple[dict[str, BaseEstimator], dict[str, BaseEstimator]]:
    r1 = {
        "Random Forest": load_model("r1_random_forest.joblib"),
        "Logistic Regression": load_model("r1_logistic_regression.joblib"),
        "MLP": load_model("r1_mlp.joblib"),
        "HistGradientBoosting": load_model("r1_hist_gradient_boosting.joblib"),
    }
    r2 = {
        "Random Forest": load_model("r2_random_forest.joblib"),
        "Logistic Regression": load_model("r2_logistic_regression.joblib"),
        "MLP": load_model("r2_mlp.joblib"),
        "HistGradientBoosting": load_model("r2_hist_gradient_boosting.joblib"),
    }
    return r1, r2