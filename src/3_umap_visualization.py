from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

from utils import FIGURES_DIR, PROCESSED_DIR, ensure_dirs


def stratified_subsample(
    x: np.ndarray, y: np.ndarray, max_points: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_points:
        return x, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_points, random_state=random_state)
    idx, _ = next(sss.split(x, y))
    return x[idx], y[idx]


def main() -> None:
    ensure_dirs()
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "UMAP requires 'umap-learn'. Install with: pip install umap-learn"
        ) from e

    x_train = np.load(PROCESSED_DIR / "X_train.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    x_plot, y_plot = stratified_subsample(x_all, y_all, max_points=20000, random_state=42)

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="euclidean", n_components=2, random_state=42
    )
    emb = reducer.fit_transform(x_plot)

    emb_df = pd.DataFrame({"umap_1": emb[:, 0], "umap_2": emb[:, 1], "cardio": y_plot})
    emb_df.to_csv(FIGURES_DIR / "3_umap_embedding_sample.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=emb_df,
        x="umap_1",
        y="umap_2",
        hue="cardio",
        alpha=0.6,
        s=10,
        palette={0: "#4C72B0", 1: "#DD8452"},
        linewidth=0,
    )
    plt.title("UMAP Projection (stratified sample)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="cardio")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "3_umap_projection.png", dpi=180)
    plt.close()

    print(f"Saved UMAP figure: {FIGURES_DIR / '3_umap_projection.png'}")
    print(f"Saved embedding data: {FIGURES_DIR / '3_umap_embedding_sample.csv'}")


if __name__ == "__main__":
    main()
