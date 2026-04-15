from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_RAW = RAW_DIR / "cardio_train.csv"


def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def locate_raw_dataset() -> Path:
    if DEFAULT_RAW.exists():
        return DEFAULT_RAW
    raise FileNotFoundError(f"Could not find cardio dataset at {DEFAULT_RAW}")


def load_raw_cardio() -> pd.DataFrame:
    return pd.read_csv(locate_raw_dataset(), sep=";")


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


def feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["cardio"]).copy()
    y = df["cardio"].astype(int).copy()
    if "id" in x.columns:
        x = x.drop(columns=["id"])
    x["age_years"] = x["age"] / 365.25
    x["bmi"] = x["weight"] / ((x["height"] / 100.0) ** 2)
    return x, y


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


def save_metrics_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    path = METRICS_DIR / filename
    df.to_csv(path, index=False)
    return path
