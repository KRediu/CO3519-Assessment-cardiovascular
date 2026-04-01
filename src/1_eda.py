from __future__ import annotations

import pandas as pd

from utils import METRICS_DIR, clean, ensure_dirs, load_cardio


def main() -> None:
    ensure_dirs()
    df = load_cardio()
    df_c = clean(df)
    report = pd.DataFrame(
        [
            {
                "stage": "raw",
                "rows": len(df),
                "duplicates": int(df.duplicated().sum()),
                "missing_total": int(df.isna().sum().sum()),
                "target_0": int((df["cardio"] == 0).sum()),
                "target_1": int((df["cardio"] == 1).sum()),
            },
            {
                "stage": "cleaned",
                "rows": len(df_c),
                "duplicates": int(df_c.duplicated().sum()),
                "missing_total": int(df_c.isna().sum().sum()),
                "target_0": int((df_c["cardio"] == 0).sum()),
                "target_1": int((df_c["cardio"] == 1).sum()),
            },
        ]
    )
    out = METRICS_DIR / "1_eda_summary.csv"
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
