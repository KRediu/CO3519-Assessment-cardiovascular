# Change python behavior
from __future__ import annotations

# General library imports
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

# Custom imports
from utils import load_processed_data, save_metrics_csv, save_model


# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Create mlp model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=1e-3,
    learning_rate_init=1e-3,
    early_stopping=True,
    max_iter=500,
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

# Save prediction results into table
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

# Print confirmation
print(f"Saved MLP metrics: {path}")

# Save trained model and print confirmation
model_path = save_model(model, "r1_mlp.joblib")
print(f"Saved MLP model: {model_path}")