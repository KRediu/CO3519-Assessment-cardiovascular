# CO3519 Assessment

## Structure

```text
CO3519-Assessment-cardiovascular/
|____ data/
|     |____ processed/
|     |____ raw/
|
|____ models/
|	  |____ kmeans.joblib
|	  |____ kmeans_pca.joblib
|	  |____ kmeans_scaler.joblib
|     |____ r1_hist_gradient_boosting.joblib
|     |____ r1_logistic_regression.joblib
|     |____ r1_mlp.joblib
|     |____ r1_random_forest.joblib
|     |____ r2_hist_gradient_boosting.joblib
|     |____ r2_logistic_regression.joblib
|     |____ r2_mlp.joblib
|     |____ r2_random_forest.joblib
|     |____ stacking.joblib
|
|____ results/
|     |____ figures/
|     |     |____ 3_umap_embedding_sample.csv
|     |     |____ 3_umap_projection.png
|     |     |____ class_balance.png
|     |     |____ confusion_matrices_r1.png
|     |     |____ confusion_matrices_r2.png
|     |     |____ confusion_matrices_stacking.png
|     |     |____ correlation_heatmap.png
|     |     |____ pr_curves_r1.png
|     |     |____ pr_curves_r2.png
|     |     |____ roc_curves_r1.png
|     |     |____ roc_curves_r2.png
|     |     |____ 13_kmeans_clusters.png
|     |     |____ 13_kmeans_selection.png
|     |     |____ 14_confusion_matrices_risk_score.png
|     |     |____ 14_pr_curves_risk_score.png
|     |     |____ 14_risk_score_distribution.png
|     |     |____ 14_roc_curves_risk_score.png
|     |
|     |____ metrics/
|           |____ 1_data_audit.csv
|           |____ 2_split_summary.csv
|           |____ 4_random_forest_metrics.csv
|           |____ 5_logistic_regression_metrics.csv
|           |____ 6_mlp_metrics.csv
|           |____ 7_hist_gradient_boosting_metrics.csv
|           |____ 8_tuned_best_params.json
|           |____ 8_tuned_thresholds.json
|           |____ 8_tuned_model_metrics.csv
|           |____ 9_stacking_metrics.csv
|           |____ 10_model_comparison.csv
|           |____ 12_round_delta_report.csv
|           |____ 13_kmeans_cluster_profile.csv
|           |____ 13_kmeans_metrics.csv
|           |____ 13_kmeans_search.csv
|           |____ 13_kmeans_search_trace.csv
|           |____ 13_kmeans_search_stability.csv
|           |____ 13_kmeans_search_vs_gmm.csv
|           |____ 14_stacking_risk_score_metrics.csv
|
|____ src/
|     |____ 1_eda.py
|     |____ 2_preprocessing.py
|     |____ 3_umap_visualization.py
|     |____ 4_random_forest.py
|     |____ 5_logistic_regression.py
|     |____ 6_mlp.py
|     |____ 7_hist_gradient_boosting.py
|     |____ 8_hyperparameter_tuning.py
|	  |____ 9_stacking.py
|     |____ 10_results_comparison.py
|     |____ 11_experimental_comparison.py
|     |____ 12_visualizations.py
|     |____ 13_kmeans.py
|     |____ 14_stacking_risk_score.py
|     |____ utils.py
|
|____ .gitignore
|____ implementation_notes.txt
|____ README.md
|____ requirements.txt
|____ run_pipeline.py
```

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Full pipeline run

```powershell
python .\run_pipeline.py
```

Useful flags:

```powershell
python .\run_pipeline.py --skip-umap
python .\run_pipeline.py --skip-visuals
python .\run_pipeline.py --skip-dependencies
```

## Manual run order

```powershell
python .\src\1_eda.py
python .\src\2_preprocessing.py
python .\src\3_umap_visualization.py
python .\src\4_random_forest.py
python .\src\5_logistic_regression.py
python .\src\6_mlp.py
python .\src\7_hist_gradient_boosting.py
python .\src\8_hyperparameter_tuning.py
python .\src\9_stacking.py
python .\src\10_results_comparison.py
python .\src\11_round_delta_report.py
python .\src\12_visualizations.py
python .\src\13_kmeans.py
```

## Outputs

- `results/metrics/`: audits, per-model metrics, tuned metrics, model comparison, round-delta report, and K-Means summaries
- `results/figures/`: UMAP projection, visualization figures, and the K-Means cluster plot
- `data/processed/`: split arrays and transformed feature names
- `models/`: all trained models for both rounds
