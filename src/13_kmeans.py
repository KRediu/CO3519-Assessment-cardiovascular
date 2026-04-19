# Change python behavior
from __future__ import annotations

# General library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Custom imports
from utils import FIGURES_DIR, METRICS_DIR, ensure_dirs, load_processed_data, save_metrics_csv, save_model


# Ensure the directories exist
ensure_dirs()

# Return a balanced-ish sample using stratification over the class labels.
def stratified_sample_indices(y: np.ndarray, max_points: int, random_state: int) -> np.ndarray:
    if len(y) <= max_points:
        return np.arange(len(y))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_points, random_state=random_state)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return idx


print("Loading processed data...")
x_train, x_test, y_train, y_test = load_processed_data()
print(f"Train shape: {x_train.shape}, test shape: {x_test.shape}")

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# PCA
pca = PCA(n_components=0.95, random_state=42)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)
print(f"PCA reduced dimensions from {x_train_scaled.shape[1]} to {x_train_pca.shape[1]}")

x_cluster = x_train_pca
x_test_cluster = x_test_pca

# Pick a small representative subset for model selection so the search stays fast.
search_idx = stratified_sample_indices(y_train, max_points=min(5000, len(y_train)), random_state=42)
x_search = x_cluster[search_idx]

candidate_ks = range(2, 7)
search_rows: list[dict] = []
inertias = []
silhouettes = []
ch_scores = []

print("Searching for the best k...")
for k in candidate_ks:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    labels = model.fit_predict(x_search)
    sil = silhouette_score(x_search, labels, sample_size=min(2000, len(x_search)), random_state=42)
    ch = calinski_harabasz_score(x_search, labels)
    inertias.append(model.inertia_)
    silhouettes.append(sil)
    ch_scores.append(ch)

    search_rows.append(
        {
            "k": k,
            "sample_silhouette": float(sil),
            "sample_inertia": float(model.inertia_),
            "calinski_harabasz": float(ch),
        }
    )
    print(f"  k={k}: silhouette={sil:.4f}, inertia={model.inertia_:.1f}, CH={ch:.1f}")

# Choose k based on silhouette (or you could use a combined rule)
best_k = candidate_ks[np.argmax(silhouettes)]
print(f"Best k selected (max silhouette): {best_k}")

# Plot elbow and silhoute for manual inspection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(candidate_ks, inertias, marker='o')
ax1.set_xlabel('k')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(candidate_ks, silhouettes, marker='o', color='green')
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "13_kmeans_selection.png", dpi=150)
plt.close()

# Fit the final model on the full training data.
final_model = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init=25,
    max_iter=300,
    random_state=42,
)
final_model.fit(x_cluster)

train_clusters = final_model.predict(x_cluster)
test_clusters = final_model.predict(x_test_cluster)

print("Evaluating final clustering model...")
train_silhouette = silhouette_score(x_cluster, train_clusters, sample_size=min(2000, len(x_cluster)), random_state=42)
train_inertia = float(final_model.inertia_)
test_ari = float(adjusted_rand_score(y_test, test_clusters))
test_nmi = float(normalized_mutual_info_score(y_test, test_clusters))

cluster_profile = (
    pd.DataFrame({"cluster": test_clusters, "cardio": y_test})
    .groupby("cluster", as_index=False)
    .agg(count=("cardio", "size"), disease_rate=("cardio", "mean"))
    .sort_values("cluster")
)
cluster_profile["disease_rate"] = cluster_profile["disease_rate"].astype(float)

summary_df = pd.DataFrame(
    [
        {
            "model": "kmeans",
            "round": "exploratory",
            "best_k": int(best_k),
            "train_silhouette": float(train_silhouette),
            "train_inertia": train_inertia,
            "test_ari": test_ari,
            "test_nmi": test_nmi,
        }
    ]
)
# Stability assessment
seeds = [42, 123, 456, 789]
ari_list = []
nmi_list = []
for seed in seeds:
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=25)
    labels = km.fit_predict(x_cluster)       # train on full PCA data
    test_labels = km.predict(x_test_cluster)
    ari_list.append(adjusted_rand_score(y_test, test_labels))
    nmi_list.append(normalized_mutual_info_score(y_test, test_labels))

stability_df = pd.DataFrame({
    "seed": seeds,
    "test_ari": ari_list,
    "test_nmi": nmi_list,
})
stability_df.loc["mean"] = stability_df.mean(numeric_only=True)
stability_df.loc["std"] = stability_df.std(numeric_only=True)

print(f"Stability over seeds: ARI mean={np.mean(ari_list):.3f} ± {np.std(ari_list):.3f}")
print(f"                      NMI mean={np.mean(nmi_list):.3f} ± {np.std(nmi_list):.3f}")

# Compare with Gaussian Mixture Model
print("Fitting GMM for comparison...")
gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42, n_init=10)
gmm.fit(x_cluster)
gmm_test_labels = gmm.predict(x_test_cluster)
gmm_ari = adjusted_rand_score(y_test, gmm_test_labels)
gmm_nmi = normalized_mutual_info_score(y_test, gmm_test_labels)
print(f"GMM on test: ARI={gmm_ari:.3f}, NMI={gmm_nmi:.3f}")

# Save comparison row
comparison_df = pd.DataFrame([
    {"model": "kmeans", "test_ari": test_ari, "test_nmi": test_nmi},
    {"model": "gmm", "test_ari": gmm_ari, "test_nmi": gmm_nmi},
])

# Save outputs
search_df = pd.DataFrame(search_rows).sort_values("sample_silhouette", ascending=False)
summary_path = save_metrics_csv(summary_df, "13_kmeans_metrics.csv")
search_path = save_metrics_csv(search_df, "13_kmeans_search.csv")
profile_path = save_metrics_csv(cluster_profile, "13_kmeans_cluster_profile.csv")
search_trace_path = save_metrics_csv(search_df, "13_kmeans_search_trace.csv")
stability_path = save_metrics_csv(stability_df, "13_kmeans_stability.csv")
comparison_path = save_metrics_csv(comparison_df, "13_kmeans_vs_gmm.csv")
model_path = save_model(final_model, "kmeans.joblib")
scaler_path = save_model(scaler, "kmeans_scaler.joblib")
pca_path = save_model(pca, "kmeans_pca.joblib")

# Visualisation
print("Building cluster plot...")
plot_idx = stratified_sample_indices(y_train, max_points=min(5000, len(y_train)), random_state=7)
x_plot = x_cluster[plot_idx]
cluster_plot = final_model.predict(x_plot)
coords = PCA(n_components=2, random_state=42).fit_transform(x_plot)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_plot, cmap="tab10", s=14, alpha=0.75)
ax.set_title(f"K-Means Clusters (k={best_k}) on a training sample (PCA projection)")
ax.set_xlabel("PCA-1")
ax.set_ylabel("PCA-2")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
ax.add_artist(legend1)
plt.tight_layout()
figure_path = FIGURES_DIR / "13_kmeans_clusters.png"
plt.savefig(figure_path, dpi=180)
plt.close()

print(f"Saved K-Means model: {model_path}")
print(f"Saved K-Means scaler: {scaler_path}")
print(f"Saved K-Means PCA: {pca_path}")
print(f"Saved K-Means summary: {summary_path}")
print(f"Saved K-Means search results: {search_path}")
print(f"Saved K-Means cluster profile: {profile_path}")
print(f"Saved K-Means search trace: {search_trace_path}")
print(f"Saved K-Means stability: {stability_path}")
print(f"Saved K-Means vs GMM comparison: {comparison_path}")
print(f"Saved K-Means figure: {figure_path}")
