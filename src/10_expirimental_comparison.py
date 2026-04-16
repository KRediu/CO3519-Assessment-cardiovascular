# Change python behavior
from __future__ import annotations

# General library imports
import pandas as pd

# Custom imports
from utils import METRICS_DIR, save_metrics_csv


# Ensure comparison data between models exists
comp_path = METRICS_DIR / "9_model_comparison.csv"
if not comp_path.exists():
    raise FileNotFoundError(
        "Missing 9_model_comparison.csv. Run 9_results_comparison.py first."
    )

# Ensure all columns exists in csv
df = pd.read_csv(comp_path)
need_cols = {"model", "round", "test_acc", "test_f1", "test_auc"}
if not need_cols.issubset(df.columns):
    raise ValueError(f"Comparison file missing required columns: {need_cols}")

# Split data per round
r1 = (
    df[df["round"] == "round_1"]
    .sort_values("test_auc", ascending=False)
    .drop_duplicates(subset=["model"], keep="first")
    .set_index("model")
)
r2 = (
    df[df["round"] == "round_2_tuned"]
    .sort_values("test_auc", ascending=False)
    .drop_duplicates(subset=["model"], keep="first")
    .set_index("model")
)

# Ensure there are models that have data in both rounds
common = sorted(set(r1.index).intersection(r2.index))
if not common:
    raise ValueError("No common models between round_1 and round_2_tuned.")

# Export comparison between round 1 and 2 per model
rows = []
for m in common:
    rows.append(
        {
            "model": m,
            "round_1_test_acc": float(r1.loc[m, "test_acc"]),
            "round_2_test_acc": float(r2.loc[m, "test_acc"]),
            "delta_test_acc": float(r2.loc[m, "test_acc"] - r1.loc[m, "test_acc"]),
            "round_1_test_f1": float(r1.loc[m, "test_f1"]),
            "round_2_test_f1": float(r2.loc[m, "test_f1"]),
            "delta_test_f1": float(r2.loc[m, "test_f1"] - r1.loc[m, "test_f1"]),
            "round_1_test_auc": float(r1.loc[m, "test_auc"]),
            "round_2_test_auc": float(r2.loc[m, "test_auc"]),
            "delta_test_auc": float(r2.loc[m, "test_auc"] - r1.loc[m, "test_auc"]),
        }
    )
out_df = pd.DataFrame(rows).sort_values("delta_test_auc", ascending=False)
out_path = save_metrics_csv(out_df, "10_round_delta_report.csv")

# Print confirmation
print(f"Saved round delta report: {out_path}")
print(out_df.to_string(index=False))