# Change python behavior
from __future__ import annotations

# Standard library imports
import json

# General library imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

# Custom imports
from utils import METRICS_DIR, load_processed_data, save_metrics_csv, save_model


# Chooses best threshold for medical ML
def best_threshold(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        score = fbeta_score(y_test, preds, beta=2, zero_division=0)
        if score > best_f2:
            best_f2, best_t = score, float(t)
    return best_t


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
    ) -> tuple[float, float, float, float, float]:
    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return (
        float(accuracy_score(y_test, pred)),
        float(f1_score(y_test, pred, zero_division=0)),
        float(roc_auc_score(y_test, proba)),
        float(recall_score(y_test, pred, zero_division=0)),
        float(fbeta_score(y_test, pred, beta=2, zero_division=0)),
    )

# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Create cross validation of 3 folds and a reproductible random generator
cv = StratifiedKFold(
    n_splits=3, 
    shuffle=True, 
    random_state=42
)
rng = np.random.RandomState(42)

# Make an f2 scoring, more suitable for medical reasons, best for less false negatives
CV_SCORING = make_scorer(fbeta_score, beta=2) 

# Compute scale_pos_weight for XGBoost to handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Create a hyperparameter search config for all 5 ML models, contains model, hyperparameter search space and number of iterations
search_space = {
    "logistic_regression": (
        LogisticRegression(
            max_iter=2500, 
            class_weight="balanced", 
            solver="lbfgs", 
            random_state=42
        ),
        {
            "C": np.logspace(-3, 1, 30),
        },
        12,
    ),
    "random_forest": (
        RandomForestClassifier(
            class_weight="balanced", 
            n_jobs=-1, 
            random_state=42
        ),
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
        HistGradientBoostingClassifier(
            class_weight="balanced", 
            random_state=42
        ),
        {
            "max_depth": [3, 4, 5, 6, 8, None],
            "min_samples_leaf": [10, 20, 30, 50, 80],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "l2_regularization": [0.0, 0.5, 1.0, 2.0, 5.0],
            "max_iter": [150, 200, 300],
        },
        14,
    ),
    "xgboost": (
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "min_child_weight": [1, 5, 10, 20, 30],
        },
        15, 
    )
}

# Initialize containers for rows and bets params
rows: list[dict] = []
best_params: dict[str, dict] = {}
best_thresholds: dict[str, float] = {}

# For each ML model, do hyperparameter turning, evaluate and save model
for model_name, (base_model, param_dist, n_iter) in search_space.items():
    # For MLP pass sample_weight
    if model_name == "mlp":
        from sklearn.utils.class_weight import compute_sample_weight
        fit_params = {
            "sample_weight": compute_sample_weight("balanced", y_train)
        }
    else:
        fit_params = {} 

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=CV_SCORING,
        n_jobs=-1,
        cv=cv,
        random_state=rng,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    
    # Find best threshold 
    threshold = best_threshold(best, x_test, y_test)
    best_thresholds[model_name] = threshold
    
    # Evaluate results
    test_acc, test_f1, test_auc, test_recall, test_f2 = evaluate(
        best, x_test, y_test, threshold=threshold
    )

    rows.append(
        {
            "model": model_name,
            "round": "round_2_tuned",
            "cv_f2_mean": float(search.best_score_),
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_auc": test_auc,
            "test_recall": test_recall,
            "test_f2": test_f2,
            "decision_threshold": threshold
        }
    )
    best_params[model_name] = {
        k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
        for k, v in search.best_params_.items()
    }
    print(f"{model_name}: cv_f2={search.best_score_:.4f} test_recall={test_recall:.4f}  test_auc={test_auc:.4f} threshold={threshold:.3f}")
    model_path = save_model(best, f"r2_{model_name}.joblib")
    print(f"Saved {model_name}: {model_path}")

# Save hyperparameter tuning results into a table and a json file
tuned_df = pd.DataFrame(rows).sort_values(["test_recall", "test_auc"], ascending=False)
csv_path = save_metrics_csv(tuned_df, "9_tuned_model_metrics.csv")
params_path = METRICS_DIR / "9_tuned_best_params.json"
params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
thresholds_path = METRICS_DIR / "9_tuned_thresholds.json"
thresholds_path.write_text(json.dumps(best_thresholds, indent=2), encoding="utf-8")

# Print confirmation
print(f"Saved tuned metrics: {csv_path}")
print(f"Saved best params: {params_path}")
print(f"Saved thresholds: {thresholds_path}")