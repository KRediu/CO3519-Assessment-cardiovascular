# Change python behavior
from __future__ import annotations

# General library imports
import pandas as pd

# Custom imports
from utils import clean_cardio, data_audit, ensure_dirs, load_raw_cardio, save_metrics_csv


# Ensure the directories exist
ensure_dirs()

# Load raw data, perform quality checks, clean the data and perform quality checks again
df = load_raw_cardio()
raw = data_audit(df)
cleaned = clean_cardio(df)
clean = data_audit(cleaned)

# Export summary of data from raw and cleaned stage
out = pd.DataFrame(
    [
        {"stage": "raw", **raw},
        {"stage": "cleaned", **clean},
        {"stage": "drop_summary", "dropped_rows_after_cleaning": int(len(df) - len(cleaned))},
    ]
)
path = save_metrics_csv(out, "1_data_audit.csv")

# Print confirmation and information regarding the data
print(f"Saved EDA audit: {path}")
print(out.to_string(index=False))