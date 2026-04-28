# Change python behavior
from __future__ import annotations

# General library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer, roc_auc_score, confusion_matrix, auc, precision_recall_curve, roc_curve, average_precision_score, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

# Custom imports
from utils import load_processed_data, save_metrics_csv, save_model, load_model, FIGURES_DIR

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

cv_scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "f2": make_scorer(fbeta_score, beta=2)
}

cv_out = cross_validate(
    model,
    x_train,
    y_train,
    cv=cv,
    scoring=cv_scoring,
    n_jobs=-1,
    return_train_score=False,
)

# Train the model
model.fit(x_train, y_train)
oof_proba = cross_val_predict(
    model, x_train, y_train, cv=5, method='predict_proba', n_jobs=-1
)[:, 1]

# Find best threshold using oof predictions
thresholds = np.linspace(0.01, 0.99, 99)
f2_scores = [fbeta_score(y_train, (oof_proba >= t).astype(int), beta=2)
             for t in thresholds]
best_threshold = thresholds[np.argmax(f2_scores)]

# Now apply this threshold on the test set
proba = model.predict_proba(x_test)[:, 1]
pred = (proba >= best_threshold).astype(int)

# Run each threshold from best to 0.5
threshold_sweep = np.arange(best_threshold, 0.51, 0.1)
preds_per_threshold = np.column_stack([
    (proba >= t).astype(int) for t in threshold_sweep
])

# Get positive counts for each
positive_counts = preds_per_threshold.sum(axis=1)

# OOF risk score
oof_preds_per_threshold = np.column_stack([
    (oof_proba >= t).astype(int) for t in threshold_sweep
])
oof_positive_counts = oof_preds_per_threshold.sum(axis=1)

# CV metrics from original run
cv_acc_mean = cv_out["test_accuracy"].mean()
cv_f1_mean  = cv_out["test_f1"].mean()
cv_f2_mean  = cv_out["test_f2"].mean()
cv_auc_mean = cv_out["test_roc_auc"].mean()

# Find best cutoff using F2
possible_cutoffs = np.arange(0, 6)  # treat as positive if count >= cutoff
oof_f2_scores = [
    fbeta_score(y_train, (oof_positive_counts >= c).astype(int), beta=2)
    for c in possible_cutoffs
]
best_cutoff_f2 = possible_cutoffs[np.argmax(oof_f2_scores)]+1

# Find best cutoff using false negative rate <=3%
target_fnr = 0.03
best_cutoff_fnr = 0  # fallback
for c in sorted(possible_cutoffs, reverse=True):  # highest first
    oof_pred = (oof_positive_counts >= c).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, oof_pred).ravel()
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    if fnr <= target_fnr:
        best_cutoff_fnr = c
        break

# function to compute metrics        
def compute_metrics(cutoff, cutoff_label):
    risk_pred = (positive_counts >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, risk_pred).ravel()

    acc = accuracy_score(y_test, risk_pred)
    f1  = f1_score(y_test, risk_pred)
    f2  = fbeta_score(y_test, risk_pred, beta=2)
    auc = roc_auc_score(y_test, positive_counts)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "model": "stacking_risk_score",
        "round": "stacking_risk_score",
        "cv_acc_mean": float(cv_acc_mean),
        "cv_f1_mean": float(cv_f1_mean),
        "cv_f2_mean": float(cv_f2_mean),
        "cv_auc_mean": float(cv_auc_mean),
        "test_acc": float(acc),
        "test_f1": float(f1),
        "test_f2": float(f2),
        "test_auc": float(auc),
        "test_sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "test_fnr": float(fnr),
        "test_ppv": float(ppv),
        "test_npv": float(npv),
        "best_cutoff": int(cutoff),
        "cutoff_method": cutoff_label,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

# Run metrics for both cutoffs
row_f2  = compute_metrics(best_cutoff_f2,  "max F2 on OOF risk scores")
row_fnr = compute_metrics(best_cutoff_fnr, f"highest cutoff with OOF FNR ≤ {target_fnr}")

out = pd.DataFrame([row_f2, row_fnr])
path = save_metrics_csv(out, "14_stacking_risk_score_metrics.csv")

# Print confirmation
print(f"Saved stacking risk score metrics: {path}")

# Save trained model and print confirmation
# Probably unneeded as it is the same model as the basic stacking
model_path = save_model(model, "stacking_risk_score.joblib")
print(f"Saved stacking model: {model_path}")

# Build confusion matrix
def build_cm_dict(y_true, positive_counts, cutoffs, labels):
    cms = {}
    for cutoff, label in zip(cutoffs, labels):
        pred = (positive_counts >= cutoff).astype(int)
        cms[label] = confusion_matrix(y_true, pred, labels=[0, 1])
    return cms
    
# All copied from 12_visualizations.py
# easier than using importlib.util  
# Create single roc curves plots 
def plot_roc(data: dict, title: str, filename: str):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, roc_auc) in data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()

# Create single pr plots
def plot_pr(data: dict, title: str, filename: str):
    plt.figure(figsize=(7, 6))
    for name, (recall, precision, pr_auc) in data.items():
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()

# Create single confusion matrix plots
def plot_confusion_matrices(data: dict, title: str, filename: str, n_cols: int = 2):
    n = len(data)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]
    fig.suptitle(title)
    for i, (name, cm) in enumerate(data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                    xticklabels=["0", "1"], yticklabels=["0", "1"])
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()    
    
# ROC and PR data for raw probability
fpr_raw, tpr_raw, _ = roc_curve(y_test, proba)
roc_auc_raw = auc(fpr_raw, tpr_raw)

precision_raw, recall_raw, _ = precision_recall_curve(y_test, proba)
pr_auc_raw = auc(recall_raw, precision_raw)

# ROC and PR data for risk score (discrete)
fpr_risk, tpr_risk, _ = roc_curve(y_test, positive_counts)
roc_auc_risk = auc(fpr_risk, tpr_risk)

precision_risk, recall_risk, _ = precision_recall_curve(y_test, positive_counts)
pr_auc_risk = auc(recall_risk, precision_risk)

# Build dicts for the plotting functions
roc_data = {
    "Stacking (raw)":     (fpr_raw, tpr_raw, roc_auc_raw),
    "Stacking (risk score)": (fpr_risk, tpr_risk, roc_auc_risk),
}

pr_data = {
    "Stacking (raw)":     (recall_raw, precision_raw, pr_auc_raw),
    "Stacking (risk score)": (recall_risk, precision_risk, pr_auc_risk),
}

# Raw stacking with its best F2 threshold
pred_raw = (proba >= best_threshold).astype(int)
cm_raw = confusion_matrix(y_test, pred_raw, labels=[0, 1])

# Risk score – two clinical cutoffs
pred_risk_fnr = (positive_counts >= best_cutoff_fnr).astype(int)
pred_risk_f2  = (positive_counts >= best_cutoff_f2).astype(int)
cm_risk_fnr = confusion_matrix(y_test, pred_risk_fnr, labels=[0, 1])
cm_risk_f2  = confusion_matrix(y_test, pred_risk_f2, labels=[0, 1])

cm_data = {
    f"Stacking (raw, thr={best_threshold:.2f})": cm_raw,
    f"Risk score (FNR≤3%, cutoff≥{best_cutoff_fnr})": cm_risk_fnr,
    f"Risk score (max F2, cutoff≥{best_cutoff_f2})": cm_risk_f2,
}

# Create all plots
plot_roc(roc_data, "ROC Curves – Stacking vs. Risk Score", "14_roc_curves_risk_score.png")
plot_pr(pr_data, "Precision-Recall Curves – Stacking vs. Risk Score", "14_pr_curves_risk_score.png")
plot_confusion_matrices(
    cm_data,
    title="Confusion Matrices – Stacking vs. Risk Score",
    filename="14_confusion_matrices_risk_score.png",
    n_cols=3
)

# Risk score distribution by actual class
risk_counts_true = positive_counts[y_test == 1]
risk_counts_false = positive_counts[y_test == 0]

plt.figure(figsize=(8, 5))
bins = np.arange(-0.5, 5.5, 1)
plt.hist(
    [risk_counts_false, risk_counts_true],
    bins=bins,
    label=["True Negative", "True Positive"],
    stacked=True,
    color=["steelblue", "darkorange"],
    edgecolor="white",
    alpha=0.85
)
plt.xlabel("Risk score (positive count)")
plt.ylabel("Number of patients")
plt.title("Risk Score Distribution by True Class")
plt.xticks(range(5))
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "14_risk_score_distribution.png", dpi=150)
plt.close()

print("Saved risk score visualizations to:", FIGURES_DIR)