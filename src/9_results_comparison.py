from __future__ import annotations

import pandas as pd

from utils import METRICS_DIR, save_metrics_csv


def main() -> None:
    files = {
        "4_random_forest_metrics.csv": "round_1",
        "5_logistic_regression_metrics.csv": "round_1",
        "6_mlp_metrics.csv": "round_1",
        "7_hist_gradient_boosting_metrics.csv": "round_1",
        "8_tuned_model_metrics.csv": "round_2_tuned",
    }
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

    summary = pd.concat(dfs, ignore_index=True).sort_values(
        ["test_auc", "test_f1"], ascending=False
    )
    out = save_metrics_csv(summary, "9_model_comparison.csv")
    print(f"Saved model comparison: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
