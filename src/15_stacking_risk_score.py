# Change python behavior
from __future__ import annotations

# General library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer, roc_auc_score, confusion_matrix, auc, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.calibration import calibration_curve
from scipy.stats import binom

# Custom imports
from utils import load_processed_data, save_metrics_csv, save_model, load_model, FIGURES_DIR

# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Base models
base_models = [
    ("hist_gb", load_model("r2_hist_gradient_boosting.joblib")),
    ("random_forest", load_model("r2_random_forest.joblib")),
    ("mlp", load_model("r2_mlp.joblib")),
    ("xgboost", load_model("r2_xgboost.joblib")),
    ("logistic_regression", load_model("r2_logistic_regression.joblib"))
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
    model, x_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
)[:, 1]

# Test set probabilities (the continuous risk score)
test_proba = model.predict_proba(x_test)[:, 1]

# CV metrics from original run
cv_acc_mean = cv_out["test_accuracy"].mean()
cv_f1_mean  = cv_out["test_f1"].mean()
cv_f2_mean  = cv_out["test_f2"].mean()
cv_auc_mean = cv_out["test_roc_auc"].mean()

# Find best threshold using oof predictions
thresholds = np.linspace(0.01, 0.99, 99)
f2_scores = [fbeta_score(y_train, (oof_proba >= t).astype(int), beta=2) for t in thresholds]
best_threshold_f2 = thresholds[np.argmax(f2_scores)]

# Find best cutoff using false negative rate <=3%
target_fnr = 0.03
best_cutoff_fnr = 0.5  # fallback
for t in thresholds[::-1]:  # scan from highest to lowest
    oof_pred = (oof_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, oof_pred).ravel()
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    if fnr <= target_fnr:
        best_threshold_fnr = t
        break


# Evaluate threshold function    
def evaluate_threshold(threshold, label):
    pred = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

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
        "test_acc": float(accuracy_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_f2": float(fbeta_score(y_test, pred, beta=2)),
        "test_auc": float(roc_auc_score(y_test, test_proba)),  # AUC of continuous score
        "test_sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "test_fnr": float(fnr),
        "test_ppv": float(ppv),
        "test_npv": float(npv),
        "threshold": float(threshold),
        "cutoff_method": label,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# Run metrics for both cutoffs
row_f2 = evaluate_threshold(best_threshold_f2, "max F2 on OOF")
row_fnr = evaluate_threshold(best_threshold_fnr, f"highest threshold with OOF FNR ≤ {target_fnr}")

row_cont = {
    "model": "stacking_risk_score",
    "round": "stacking_risk_score",
    "cv_acc_mean": float(cv_acc_mean),
    "cv_f1_mean": float(cv_f1_mean),
    "cv_f2_mean": float(cv_f2_mean),
    "cv_auc_mean": float(cv_auc_mean),
    "test_auc": float(roc_auc_score(y_test, test_proba)),
    "cutoff_method": "continuous risk score (AUC only)",
    "threshold": None,
}

# Per-patient risk score
risk_df = pd.DataFrame(
    {
        "y_true": y_test,
        "risk_score": test_proba,
    }
)
risk_path = save_metrics_csv(risk_df, "15_patient_risk_scores.csv")
print(f"Saved per patient risk scores: {risk_path}")

out = pd.DataFrame([row_f2, row_fnr, row_cont])
path = save_metrics_csv(out, "15_stacking_risk_score_metrics.csv")

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
    
# All copied from 13_visualizations.py
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

# ROC and PR curves for the continuous risk score (the stacking model)
fpr, tpr, _ = roc_curve(y_test, test_proba)
roc_auc_val = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, test_proba)
pr_auc_val = auc(recall, precision)

roc_data = {"Stacking (continuous risk score)": (fpr, tpr, roc_auc_val)}
pr_data = {"Stacking (continuous risk score)": (recall, precision, pr_auc_val)}

# Confusion matrices for the two clinical thresholds
cm_f2 = confusion_matrix(y_test, (test_proba >= best_threshold_f2).astype(int), labels=[0, 1])
cm_fnr = confusion_matrix(y_test, (test_proba >= best_threshold_fnr).astype(int), labels=[0, 1])

cm_data = {
    f"Threshold = {best_threshold_f2:.2f} (max F2)": cm_f2,
    f"Threshold = {best_threshold_fnr:.2f} (FNR ≤ {target_fnr})": cm_fnr,
}

# Create all plots
plot_roc(roc_data, "ROC Curve – Stacking Risk Score", "15_roc_curve_risk_score.png")
plot_pr(pr_data, "Precision‑Recall Curve – Stacking Risk Score", "15_pr_curve_risk_score.png")
plot_confusion_matrices(
    cm_data,
    title="Confusion Matrices – Two Clinical Thresholds",
    filename="15_confusion_matrices_risk_score.png",
    n_cols=2,
)

# Distribution of risk scores split by true class
plt.figure(figsize=(8, 5))
plt.hist(
    test_proba[y_test == 0],
    bins=30,
    alpha=0.6,
    color="steelblue",
    edgecolor="white",
    label="Actual Negative",
)
plt.hist(
    test_proba[y_test == 1],
    bins=30,
    alpha=0.6,
    color="darkorange",
    edgecolor="white",
    label="Actual Positive",
)
plt.xlabel("Predicted probability (risk score)")
plt.ylabel("Number of patients")
plt.title("Distribution of Risk Scores by True Class")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "15_risk_score_distribution.png", dpi=150)
plt.close()

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='Stacking Risk Score')
plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "15_calibration_curve.png", dpi=150)
plt.close()

# Decision curve
def net_benefit(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

thresholds = np.linspace(0.01, 0.99, 99)
nb_model = [net_benefit(y_test, test_proba, t) for t in thresholds]
nb_all   = [np.mean(y_test) - (1 - np.mean(y_test)) * (t / (1 - t)) for t in thresholds]

plt.figure(figsize=(8,6))
plt.plot(thresholds, nb_model, label='Stacking risk score')
plt.plot(thresholds, nb_all,   label='Treat all', linestyle='--')
plt.plot(thresholds, [0]*len(thresholds), label='Treat none', linestyle=':')
plt.xlabel('Threshold probability')
plt.ylabel('Net benefit')
plt.title('Decision Curve')
plt.legend()
plt.ylim(-0.1, 0.5)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "15_decision_curve.png", dpi=150)
plt.close()

# Risk stratification table/plot
n_bins = 10
test_df_plot = pd.DataFrame({'risk': test_proba, 'cardio': y_test})
test_df_plot['decile'] = pd.qcut(test_df_plot['risk'], q=n_bins, labels=False, duplicates='drop')
summary = test_df_plot.groupby('decile').agg(
    mean_risk=('risk', 'mean'),
    event_rate=('cardio', 'mean'),
    count=('cardio', 'count')
).reset_index()

summary['ci_low'] = [binom.interval(0.95, n, p)[0]/n for n, p in zip(summary['count'], summary['event_rate'])]
summary['ci_high'] = [binom.interval(0.95, n, p)[1]/n for n, p in zip(summary['count'], summary['event_rate'])]

plt.figure(figsize=(8,5))
plt.bar(summary['decile'], summary['event_rate'], yerr=[summary['event_rate']-summary['ci_low'], summary['ci_high']-summary['event_rate']],
        capsize=4, color='steelblue', alpha=0.7)
plt.xlabel('Risk decile (1=lowest)')
plt.ylabel('Observed disease rate')
plt.title('Risk Stratification by Decile')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "15_risk_decile_plot.png", dpi=150)
plt.close()

# Lift / cumulative gains chart
order = np.argsort(-test_proba)
pos = y_test[order]
cum_pos = np.cumsum(pos) / np.sum(pos)
cum_total = np.arange(1, len(pos)+1) / len(pos)

plt.figure(figsize=(6,6))
plt.plot(cum_total, cum_pos, label='Stacking risk score (lift)')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('Fraction of population screened')
plt.ylabel('Fraction of positives detected')
plt.title('Cumulative Gains Curve')
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "15_cumulative_gains.png", dpi=150)
plt.close()

print(f"Saved risk score visualizations to: {FIGURES_DIR}")