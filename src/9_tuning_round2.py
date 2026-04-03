from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from utils import METRICS_DIR, PROCESSED_DIR, ensure_dirs, save_metrics_csv


def main() -> None:
    ensure_dirs()
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    base = HistGradientBoostingClassifier(
        max_depth=6,
        min_samples_leaf=30,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    base.fit(x_train, y_train)
    base_proba = base.predict_proba(x_test)[:, 1]
    default_run = {
        "stage": "hgb_default",
        "test_acc": float(accuracy_score(y_test, base.predict(x_test))),
        "test_f1": float(f1_score(y_test, base.predict(x_test))),
        "test_auc": float(roc_auc_score(y_test, base_proba)),
        "test_ap": float(average_precision_score(y_test, base_proba)),
    }

    param_dist = {
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [10, 20, 30, 50],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "l2_regularization": [0.1, 1.0, 5.0],
        "max_leaf_nodes": [31, 63, 127, None],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=24,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    pred = best.predict(x_test)
    proba = best.predict_proba(x_test)[:, 1]
    tuned = {
        "stage": "hgb_after_random_search",
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_auc": float(roc_auc_score(y_test, proba)),
        "test_ap": float(average_precision_score(y_test, proba)),
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": str(search.best_params_),
    }

    pd.DataFrame([default_run, tuned]).to_csv(METRICS_DIR / "9_tuning_round2.csv", index=False)
    print(pd.DataFrame([default_run, tuned]).to_string(index=False))
    print(f"Saved: {METRICS_DIR / '9_tuning_round2.csv'}")

    tuned_metrics = pd.DataFrame(
        [
            {
                "model": "hist_gradient_boosting_tuned",
                "cv_acc_mean": float("nan"),
                "cv_f1_mean": float("nan"),
                "cv_auc_mean": float(search.best_score_),
                "test_acc": float(accuracy_score(y_test, pred)),
                "test_f1": float(f1_score(y_test, pred)),
                "test_auc": float(roc_auc_score(y_test, proba)),
            }
        ]
    )
    path = save_metrics_csv(tuned_metrics, "9_tuned_model_metrics.csv")
    print(f"Saved tuned model row for comparison: {path}")


if __name__ == "__main__":
    main()
