# Change python behavior
from __future__ import annotations

# General library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Custom imports
from utils import FIGURES_DIR, load_processed_data, ensure_dirs, stratified_subsample


# Ensure the directories exist
ensure_dirs()

# User friendly error message for missing dependency
try:
    import umap
except ImportError as e:
    raise ImportError(
        "UMAP requires 'umap-learn'. Install with: pip install umap-learn"
    ) from e

# Load train and test data
x_train, x_test, y_train, y_test = load_processed_data()

# Combine train and test data, then take a stratified sample for visualisation
x_all = np.vstack([x_train, x_test])
y_all = np.concatenate([y_train, y_test])
x_plot, y_plot = stratified_subsample(x_all, y_all, max_points=20000, random_state=42)

# Create 2D visualisation using UMAP
reducer = umap.UMAP(
    n_neighbors=30, 
    min_dist=0.1,
    metric="euclidean", 
    n_components=2, 
    n_jobs=-1,
    random_state=42
)
emb = reducer.fit_transform(x_plot)

# Convert UMAP embedding into a tables
emb_df = pd.DataFrame({"umap_1": emb[:, 0], "umap_2": emb[:, 1], "cardio": y_plot})
emb_df.to_csv(FIGURES_DIR / "3_umap_embedding_sample.csv", index=False)

# Save UMAP embedding as a plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=emb_df,
    x="umap_1",
    y="umap_2",
    hue="cardio",
    alpha=0.6,
    s=10,
    palette={0: "#4C72B0", 1: "#DD8452"},
    linewidth=0
)
plt.title("UMAP Projection (stratified sample)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="cardio")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "3_umap_projection.png", dpi=180)
plt.close()

# Print confirmation and umap dataset
print(f"Saved UMAP figure: {FIGURES_DIR / '3_umap_projection.png'}")
print(f"Saved embedding data: {FIGURES_DIR / '3_umap_embedding_sample.csv'}")