# Change python behavior
from __future__ import annotations

# General library imports
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

# Custom imports
from utils import load_processed_data, save_metrics_csv, save_model, load_model

# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Base models
base_models = [
    ("hist_gb", load_model("r2_hist_gradient_boosting.joblib")),
    ("random_forest", load_model("r2_random_forest.joblib")),
    ("mlp", load_model("r2_mlp.joblib")),
]

# Build stacking ensemble
meta_learner = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

# Cross validate on training set
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_out = cross_validate(
    model,
    x_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "f1", "roc_auc"],
    n_jobs=-1,
    return_train_score=False,
)

# Train the model
model.fit(x_train, y_train)

# Find optimal threshold
train_proba = model.predict_proba(x_train)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = [f1_score(y_train, (train_proba >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]

# Use optimal threshold
proba = model.predict_proba(x_test)[:, 1]
pred = (proba >= best_threshold).astype(int)

# Save prediction results into table
out = pd.DataFrame(
    [
        {
            "model": "stacking",
            "round": "stacking_ensemble",
            "cv_acc_mean": float(cv_out["test_accuracy"].mean()),
            "cv_f1_mean": float(cv_out["test_f1"].mean()),
            "cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
            "test_acc": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_auc": float(roc_auc_score(y_test, proba)),
        }
    ]
)
path = save_metrics_csv(out, "9_stacking_metrics.csv")

# Print confirmation
print(f"Saved stacking metrics: {path}")

# Save trained model and print confirmation
model_path = save_model(model, "stacking.joblib")
print(f"Saved stacking model: {model_path}")
