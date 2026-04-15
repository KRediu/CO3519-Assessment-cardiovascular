from __future__ import annotations

import pandas as pd

from utils import clean_cardio, data_audit, ensure_dirs, load_raw_cardio, save_metrics_csv


def main() -> None:
    ensure_dirs()
    df = load_raw_cardio()
    raw = data_audit(df)
    cleaned = clean_cardio(df)
    clean = data_audit(cleaned)

    out = pd.DataFrame(
        [
            {"stage": "raw", **raw},
            {"stage": "cleaned", **clean},
            {"stage": "drop_summary", "dropped_rows_after_cleaning": int(len(df) - len(cleaned))},
        ]
    )
    path = save_metrics_csv(out, "1_data_audit.csv")
    print(f"Saved EDA audit: {path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
