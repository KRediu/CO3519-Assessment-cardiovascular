from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

def ensure_dirs() -> None:
    for d in [RAW_DIR, PROCESSED_DIR, METRICS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_cardio() -> pd.DataFrame:
    path = RAW_DIR / "cardio_train.csv"
    if not path.exists():
        raise FileNotFoundError("Place cardio_train.csv in data/raw.")
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> pd.DataFrame:
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


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["cardio"]).copy()
    y = df["cardio"].astype(int).copy()
    if "id" in x.columns:
        x = x.drop(columns=["id"])
    x["age_years"] = x["age"] / 365.25
    x["bmi"] = x["weight"] / ((x["height"] / 100.0) ** 2)
    return x, y


def preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    num = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = [c for c in x_train.columns if c not in num]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]
                ),
                cat,
            ),
        ]
    )


def save_np(name: str, arr: np.ndarray) -> None:
    np.save(PROCESSED_DIR / f"{name}.npy", arr)


def save_metrics_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    path = METRICS_DIR / filename
    df.to_csv(path, index=False)
    return path
