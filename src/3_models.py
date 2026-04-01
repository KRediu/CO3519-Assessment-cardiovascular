from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from utils import METRICS_DIR, PROCESSED_DIR


def evaluate(model, x_train, y_train, x_test, y_test) -> dict:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_out = cross_validate(model, x_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return {
        "cv_acc_mean": float(cv_out["test_accuracy"].mean()),
        "cv_f1_mean": float(cv_out["test_f1"].mean()),
        "cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_auc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=15, class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }

    rows = []
    for name, model in models.items():
        m = evaluate(model, x_train, y_train, x_test, y_test)
        rows.append({"model": name, **m})

    out_df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    out = METRICS_DIR / "3_model_results.csv"
    out_df.to_csv(out, index=False)
    print(out_df.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
