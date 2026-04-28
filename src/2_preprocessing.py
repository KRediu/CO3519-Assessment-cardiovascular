# Change python behavior
from __future__ import annotations

# General library imports
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom imports
from utils import PROCESSED_DIR, build_preprocessor, clean_cardio, ensure_dirs, feature_engineer_clean, load_raw_cardio, save_metrics_csv


# Ensure the directories exist
ensure_dirs()

# Load raw data and split features and target
df = clean_cardio(load_raw_cardio())
x, y = feature_engineer_clean(df)

# Split dataset into training and testing data (80/20)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# DBSCAN filter on train data only
dbscan_numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
dbscan_scaled_train = StandardScaler().fit_transform(x_train[dbscan_numeric_cols])
dbscan = DBSCAN(eps=2.0, min_samples=40, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(dbscan_scaled_train)
dbscan_keep_mask = dbscan_labels != -1

x_train = x_train.iloc[dbscan_keep_mask].copy()
y_train = y_train.iloc[dbscan_keep_mask].copy()
dbscan_removed_rows = int((~dbscan_keep_mask).sum())

# Preprocess training and test data for ML
pre = build_preprocessor(x_train)
x_train_t = pre.fit_transform(x_train)
x_test_t = pre.transform(x_test)

# SMOTE creates new minority-class samples so the model sees a balanced training distribution
smote = SMOTE(random_state=42, k_neighbors=5)
x_train_t, y_train_resampled = smote.fit_resample(x_train_t, y_train)
y_train = pd.Series(y_train_resampled) 

# Save ML data as .npy files for faster and more efficient save/load
np.save(PROCESSED_DIR / "X_train.npy", x_train_t)
np.save(PROCESSED_DIR / "X_test.npy", x_test_t)
np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

# Export preprocessed data
cols = pre.get_feature_names_out()
pd.Series(cols, name="feature").to_csv(PROCESSED_DIR / "feature_names.csv", index=False)
save_metrics_csv(
    pd.DataFrame(
        [
            {"split": "train", "rows": len(y_train), "positive_rate": float(y_train.mean()), "dbscan_removed_rows": dbscan_removed_rows},
            {"split": "test", "rows": len(y_test), "positive_rate": float(y_test.mean())},
        ]
    ),
    "2_split_summary.csv",
)

# Print confirmation
print(f"Saved processed arrays in {PROCESSED_DIR}")