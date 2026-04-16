# Change python behavior
from __future__ import annotations

# General library imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# Custom imports
from utils import FIGURES_DIR, load_processed_data, clean_cardio, ensure_dirs, load_raw_cardio, load_models_by_round


# Evaluate the trained models on same data
def evaluate_models(models, x_test, y_test):
    roc_data = {}
    pr_data = {}
    cms = {}

    for name, model in models.items():
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        pr_data[name] = (recall, precision, auc(recall, precision))
        cms[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return roc_data, pr_data, cms


# Create single class balance plot
def plot_class_balance() -> None:
    df = clean_cardio(load_raw_cardio())
    counts = df["cardio"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=["No disease (0)", "Disease (1)"],
        y=counts.values,
        color="#4C72B0",
    )
    plt.title("Class Balance (Cleaned Data)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close()


# Create single correlation heatmap plot
def plot_correlation_heatmap() -> None:
    df = clean_cardio(load_raw_cardio()).copy()
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    cols = [
        "age_years",
        "height",
        "weight",
        "bmi",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "cardio",
    ]
    corr = df[cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True)
    plt.title("Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


# Create single roc curves plots 
def plot_roc(items, title, filename):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, roc_auc) in items:
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
def plot_roc(items, title, filename):
    plt.figure(figsize=(7, 6))
    for name, (recall, precision, pr_auc) in items:
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()



# Create single confusion matrix plots
def plot_confusion_matrices(cms, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    fig.suptitle(title)
    for i, (name, cm) in enumerate(cms):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


# Create all r1 & r2 plots (roc curves, precision–recall curves, and confusion matrices)
def plots_per_round() -> None:
    # Load data and models
    _, x_test, _, y_test = load_processed_data()
    r1_models, r2_models = load_models_by_round()

    # Evaluate both rounds
    roc_r1, pr_r1, cm_r1 = evaluate_models(r1_models, x_test, y_test)
    roc_r2, pr_r2, cm_r2 = evaluate_models(r2_models, x_test, y_test)

    # Create ROC curves plot round 1 & 2
    plot_roc(roc_r1.items(), "ROC Curves - Round 1", "roc_curves_r1.png")
    plot_roc(roc_r2.items(), "ROC Curves - Round 2", "roc_curves_r2.png")

    # Create precision–recall curves plot round 1 & 2
    plot_roc(pr_r1.items(), "Precision-Recall Curves - Round 1", "pr_curves_r1.png")
    plot_roc(pr_r2.items(), "Precision-Recall Curves - Round 2", "pr_curves_r2.png")

    # Create confusion matrices plots round 1 & 2
    plot_confusion_matrices(cm_r1.items(), "Confusion Matrices - Round 1", "confusion_matrices_r1.png")
    plot_confusion_matrices(cm_r2.items(), "Confusion Matrices - Round 2", "confusion_matrices_r2.png")


# Ensure the directories exist
ensure_dirs()

# Create and save plots
plot_class_balance()
plot_correlation_heatmap()
plots_per_round()

# Print confirmation
print(f"Saved visualizations to: {FIGURES_DIR}")