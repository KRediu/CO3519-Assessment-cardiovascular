# Change python behavior
from __future__ import annotations

# General library imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

# Custom imports
from utils import load_processed_data, save_metrics_csv


# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Create random forest model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
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

# Train the model and predict
model.fit(x_train, y_train)
pred = model.predict(x_test)
proba = model.predict_proba(x_test)[:, 1]

# Export predection results into table
out = pd.DataFrame(
    [
        {
            "model": "random_forest",
            "cv_acc_mean": float(cv_out["test_accuracy"].mean()),
            "cv_f1_mean": float(cv_out["test_f1"].mean()),
            "cv_auc_mean": float(cv_out["test_roc_auc"].mean()),
            "test_acc": float(accuracy_score(y_test, pred)),
            "test_f1": float(f1_score(y_test, pred)),
            "test_auc": float(roc_auc_score(y_test, proba)),
        }
    ]
)
path = save_metrics_csv(out, "4_random_forest_metrics.csv")

# Print confirmation
print(f"Saved RF metrics: {path}")