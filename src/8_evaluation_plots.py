from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier

from utils import FIGURES_DIR, PROCESSED_DIR, ensure_dirs


def build_models():
    return [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=15,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,
                learning_rate_init=1e-3,
                early_stopping=True,
                max_iter=200,
                random_state=42,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_depth=6,
                min_samples_leaf=30,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]


def main() -> None:
    ensure_dirs()
    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    models = build_models()
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for name, est in models:
        est.fit(x_train, y_train)
        proba = est.predict_proba(x_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax_roc, name=name.replace("_", " "))
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax_pr, name=name.replace("_", " "))

    ax_roc.set_title("ROC curves (hold-out test set)")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax_roc.legend(loc="lower right", fontsize=8)
    fig_roc.tight_layout()
    roc_path = FIGURES_DIR / "roc_curves.png"
    fig_roc.savefig(roc_path, dpi=150)
    plt.close(fig_roc)

    ax_pr.set_title("Precision–recall curves (hold-out test set)")
    ax_pr.legend(loc="upper right", fontsize=8)
    fig_pr.tight_layout()
    pr_path = FIGURES_DIR / "pr_curves.png"
    fig_pr.savefig(pr_path, dpi=150)
    plt.close(fig_pr)

    n = len(models)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig_cm, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, est) in zip(axes_flat, models, strict=False):
        est.fit(x_train, y_train)
        pred = est.predict(x_test)
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name.replace("_", " "))
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig_cm.suptitle("Confusion matrices (test set)", y=1.02)
    fig_cm.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrices.png"
    fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    print(f"Saved: {roc_path}\nSaved: {pr_path}\nSaved: {cm_path}")


if __name__ == "__main__":
    main()
