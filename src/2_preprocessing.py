from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import METRICS_DIR, PROCESSED_DIR, clean, ensure_dirs, load_cardio, make_xy, preprocessor, save_np


def main() -> None:
    ensure_dirs()
    df = clean(load_cardio())
    x, y = make_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    pre = preprocessor(x_train)
    x_train_t = pre.fit_transform(x_train)
    x_test_t = pre.transform(x_test)

    save_np("X_train", x_train_t)
    save_np("X_test", x_test_t)
    save_np("y_train", y_train.to_numpy())
    save_np("y_test", y_test.to_numpy())

    pd.Series(pre.get_feature_names_out(), name="feature").to_csv(
        PROCESSED_DIR / "feature_names.csv", index=False
    )
    pd.DataFrame(
        [
            {"split": "train", "rows": len(y_train), "positive_rate": float(y_train.mean())},
            {"split": "test", "rows": len(y_test), "positive_rate": float(y_test.mean())},
        ]
    ).to_csv(METRICS_DIR / "2_split_summary.csv", index=False)
    print("Saved processed arrays + split summary.")


if __name__ == "__main__":
    main()
