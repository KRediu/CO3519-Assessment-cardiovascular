# CO3519 Assessment

## Quick Setup + Run

Run from this project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python .\src\1_eda.py
python .\src\2_preprocessing.py
python .\src\3_models.py
```

This implementation includes EDA, preprocessing, and two of the models which we will use: Logistic Regression and Random Forest. 
HistGradientBoosting and MLP to be added.

## Structure

```text
CO3519-Assessment-main
├── requirements.txt
├── src/
│   ├── 1_eda.py
│   ├── 2_preprocessing.py
│   ├── 3_models.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
    └── metrics/
```
