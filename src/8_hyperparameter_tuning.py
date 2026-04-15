from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from utils import METRICS_DIR, PROCESSED_DIR, save_metrics_csv


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return (
        float(accuracy_score(y_test, pred)),
        float(f1_score(y_test, pred)),
        float(roc_auc_score(y_test, proba)),
    )


def main() -> None:
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rng = np.random.RandomState(42)

    search_space = {
        "logistic_regression": (
            LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs", random_state=42),
            {
                "C": np.logspace(-3, 1, 30),
            },
            12,
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
            {
                "n_estimators": [200, 300, 400, 500],
                "max_depth": [6, 8, 10, 12, None],
                "min_samples_leaf": [5, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "max_features": ["sqrt", "log2", None],
            },
            14,
        ),
        "mlp": (
            MLPClassifier(
                early_stopping=True,
                random_state=42,
                max_iter=250,
            ),
            {
                "hidden_layer_sizes": [(64, 32), (96, 48), (128, 64), (64,)],
                "alpha": [1e-5, 1e-4, 1e-3, 5e-3, 1e-2],
                "learning_rate_init": [5e-4, 1e-3, 2e-3, 5e-3],
            },
            10,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=42),
            {
                "max_depth": [3, 4, 5, 6, 8, None],
                "min_samples_leaf": [10, 20, 30, 50, 80],
                "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
                "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0],
                "max_iter": [150, 200, 300],
            },
            14,
        ),
    }

    rows: list[dict] = []
    best_params: dict[str, dict] = {}

    for model_name, (base_model, param_dist, n_iter) in search_space.items():
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            random_state=rng,
            refit=True,
            verbose=0,
        )
        search.fit(x_train, y_train)
        best = search.best_estimator_
        test_acc, test_f1, test_auc = evaluate(best, x_test, y_test)
        rows.append(
            {
                "model": model_name,
                "round": "round_2_tuned",
                "cv_acc_mean": np.nan,
                "cv_f1_mean": np.nan,
                "cv_auc_mean": float(search.best_score_),
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc,
            }
        )
        best_params[model_name] = {
            k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in search.best_params_.items()
        }
        print(f"{model_name}: best cv auc={search.best_score_:.4f}, test auc={test_auc:.4f}")

    tuned_df = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
    csv_path = save_metrics_csv(tuned_df, "8_tuned_model_metrics.csv")
    params_path = METRICS_DIR / "8_tuned_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"Saved tuned metrics: {csv_path}")
    print(f"Saved best params: {params_path}")


if __name__ == "__main__":
    main()
