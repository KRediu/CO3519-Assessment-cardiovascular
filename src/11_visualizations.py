from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

from utils import FIGURES_DIR, PROCESSED_DIR, clean_cardio, ensure_dirs, load_raw_cardio


def model_dict() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            learning_rate_init=1e-3,
            early_stopping=True,
            max_iter=200,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6,
            min_samples_leaf=30,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def plot_class_balance() -> None:
    df = clean_cardio(load_raw_cardio())
    counts = df["cardio"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["No disease (0)", "Disease (1)"], y=counts.values, color="#4C72B0")
    plt.title("Class Balance (Cleaned Data)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close()


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
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True)
    plt.title("Correlation Heatmap (Selected Features)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def plot_roc_pr_and_confusion() -> None:
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    models = model_dict()
    roc_data = {}
    pr_data = {}
    cms = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))
        pr_data[name] = (recall, precision, auc(recall, precision))
        cms[name] = confusion_matrix(y_test, y_pred, labels=[1, 0])

    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, (recall, precision, pr_auc) in pr_data.items():
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, (name, cm) in enumerate(cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["1", "0"],
            yticklabels=["1", "0"],
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].xaxis.set_label_position("top")
        axes[i].xaxis.tick_top()
        axes[i].tick_params(axis="x", bottom=False, top=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    plot_class_balance()
    plot_correlation_heatmap()
    plot_roc_pr_and_confusion()
    print(f"Saved visualizations to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
