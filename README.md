# CO3519 Assessment

## Quick setup and run

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python .\src\1_eda.py
python .\src\2_preprocessing.py
python .\src\3_random_forest.py
python .\src\4_logistic_regression.py
python .\src\5_mlp.py
python .\src\6_hist_gradient_boosting.py
python .\src\7_results_comparison.py
python .\src\8_evaluation_plots.py
python .\src\9_tuning_round2.py
```

After tuning, run `7_results_comparison.py` again if you want the tuned row in `7_model_comparison.csv`.

**Pipeline:** EDA → preprocessing → one script per model (metrics CSV each) → merge comparison → ROC / PR / confusion figures → HGB `RandomizedSearchCV` (same setup as script 6, then a search over settings) + optional tuned row for comparison.

## Structure

```text
CO3519-Assessment-main
├── requirements.txt
├── src/
│   ├── 1_eda.py
│   ├── 2_preprocessing.py
│   ├── 3_random_forest.py
│   ├── 4_logistic_regression.py
│   ├── 5_mlp.py
│   ├── 6_hist_gradient_boosting.py
│   ├── 7_results_comparison.py
│   ├── 8_evaluation_plots.py
│   ├── 9_tuning_round2.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── metrics/
    └── figures/
```
