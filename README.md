# CO3519 Assessment

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
