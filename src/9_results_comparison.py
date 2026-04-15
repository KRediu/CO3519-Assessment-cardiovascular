# Change python behavior
from __future__ import annotations

# General library imports
import pandas as pd

# Custom imports
from utils import METRICS_DIR, save_metrics_csv

# Define results for all ML models round 1 (base) and 2 (tuning)
files = {
    "4_random_forest_metrics.csv": "round_1",
    "5_logistic_regression_metrics.csv": "round_1",
    "6_mlp_metrics.csv": "round_1",
    "7_hist_gradient_boosting_metrics.csv": "round_1",
    "8_tuned_model_metrics.csv": "round_2_tuned",
}

# Load results and ensure round column exists
dfs = []
for f, default_round in files.items():
    p = METRICS_DIR / f
    if p.exists():
        df = pd.read_csv(p)
        if "round" not in df.columns:
            df["round"] = default_round
        dfs.append(df)
if not dfs:
    raise FileNotFoundError("No model metric files found. Run scripts 4-7 first.")

# Merge all results into a table and export them
summary = pd.concat(dfs, ignore_index=True).sort_values(
    ["test_auc", "test_f1"], ascending=False
)
out = save_metrics_csv(summary, "9_model_comparison.csv")

# Print confirmation
print(f"Saved model comparison: {out}")
print(summary.to_string(index=False))