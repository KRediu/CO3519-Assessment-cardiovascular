# CO3519 Assessment

## Structure

```text
CO3519-Assessment-cardiovascular/
|____ data/
|     |____ processed/
|     |____ raw/
|
|____ models/
|     |____ r1_hist_gradient_boosting.joblib
|     |____ r1_logistic_regression.joblib
|     |____ r1_mlp.joblib
|     |____ r1_random_forest.joblib
|     |____ r2_hist_gradient_boosting.joblib
|     |____ r2_logistic_regression.joblib
|     |____ r2_mlp.joblib
|     |____ r2_random_forest.joblib
|
|____ results/
|     |____ figures/
|     |     |____ 3_umap_embedding_sample.csv
|     |     |____ 3_umap_projection.png
|     |     |____ class_balance.png
|     |     |____ confusion_matrices_r1.png
|     |     |____ confusion_matrices_r2.png
|     |     |____ correlation_heatmap.png
|     |     |____ pr_curves_r1.png
|     |     |____ pr_curves_r2.png
|     |     |____ roc_curves_r1.png
|     |     |____ roc_curves_r2.png
|     |
|     |____ metrics/
|           |____ 1_data_audit.csv
|           |____ 2_split_summary.csv
|           |____ 4_random_forest_metrics.csv
|           |____ 5_logistic_regression_metrics.csv
|           |____ 6_mlp_metrics.csv
|           |____ 7_hist_gradient_boosting_metrics.csv
|           |____ 8_tuned_best_params.json
|           |____ 8_tuned_model_metrics.csv
|           |____ 9_model_comparison.csv
|           |____ 10_round_delta_report.csv
|
|____ src/
|     |____ __pycache__/
|     |____ 1_eda.py
|     |____ 2_preprocessing.py
|     |____ 3_umap_visualization.py
|     |____ 4_random_forest.py
|     |____ 5_logistic_regression.py
|     |____ 6_mlp.py
|     |____ 7_hist_gradient_boosting.py
|     |____ 8_hyperparameter_tuning.py
|     |____ 9_results_comparison.py
|     |____ 10_experimental_comparison.py
|     |____ 11_visualizations.py
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
python .\src\9_results_comparison.py
python .\src\10_round_delta_report.py
python .\src\11_visualizations.py
```

## Outputs

- `results/metrics/`: audits, per-model metrics, tuned metrics, model comparison, round-delta report
- `results/figures/`: UMAP projection and visualization figures
- `data/processed/`: split arrays and transformed feature names
- `models/`: all trained models for both rounds
