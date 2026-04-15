from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from utils import PROCESSED_DIR, save_metrics_csv


def main() -> None:
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-3,
        learning_rate_init=1e-3,
        early_stopping=True,
        max_iter=200,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_out = cross_validate(
        model, x_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    out = pd.DataFrame(
        [
            {
                "model": "mlp",
                "cv_acc_mean": float(cv_out["test_accuracy"].mean()),
                "cv_f1_mean": float(cv_out["test_f1"].mean()),
                "cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
                "test_acc": float(accuracy_score(y_test, pred)),
                "test_f1": float(f1_score(y_test, pred)),
                "test_auc": float(roc_auc_score(y_test, proba)),
            }
        ]
    )
    path = save_metrics_csv(out, "6_mlp_metrics.csv")
    print(f"Saved MLP metrics: {path}")


if __name__ == "__main__":
    main()
