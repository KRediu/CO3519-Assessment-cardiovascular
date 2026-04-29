# General library imports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

# Custom imports
from utils import load_processed_data, save_metrics_csv, save_model

# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Create XGBoost model
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    reg_lambda=1.0,          # L2 regularization (similar to l2_regularization)
    min_child_weight=30,     # proxy for min_samples_leaf (higher value = more conservative)
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0              # suppress training messages
)

# Create cross validation of 5 folds and run it
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
    n_jobs=-1
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
            "model": "xgboost",
            "cv_acc_mean": float(cv_out["test_accuracy"].mean()),
            "cv_f1_mean": float(cv_out["test_f1"].mean()),
            "cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
            "test_acc": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_auc": float(roc_auc_score(y_test, proba)),
        }
    ]
)
path = save_metrics_csv(out, "8_xgboost_metrics.csv")

# Print confirmation
print(f"Saved XGBoost metrics: {path}")

# Save trained model and print confirmation
model_path = save_model(model, "r1_xgboost.joblib")
print(f"Saved XGBoost model: {model_path}")