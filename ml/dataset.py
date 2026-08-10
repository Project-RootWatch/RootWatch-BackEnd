"""
Turns the flat readings table into the windowed sequences an LSTM trains
on, with a leakage-safe chronological split and per-feature scaling.

Core idea: slide a fixed-length window across the time series. Each
position gives one training example:
    X = the last WINDOW steps of [soil_moisture, temperature, light_level]
    y = the next HORIZON steps of soil_moisture

Splitting happens on RAW DAYS first (train/val/test never overlap in
time), and windows are only ever built within one split — never across
the boundary between them. Scalers are fit on train only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "synthetic_readings.csv"
FEATURE_COLUMNS = ["soil_moisture", "temperature", "light_level"]
TARGET_COLUMN = "soil_moisture"

WINDOW = 96   # 24h of history at 15-min steps
HORIZON = 24  # predict 6h ahead
STRIDE = 1    # slide the window 1 step at a time (maximizes examples)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 is test


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def chronological_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def fit_scalers(train_df):
    """One StandardScaler per feature, fit on training data only."""
    scalers = {}
    for col in FEATURE_COLUMNS:
        scaler = StandardScaler()
        scaler.fit(train_df[[col]].values)
        scalers[col] = scaler
    return scalers


def apply_scalers(df, scalers):
    scaled = df.copy()
    for col in FEATURE_COLUMNS:
        scaled[col] = scalers[col].transform(df[[col]].values).flatten()
    return scaled


def make_windows(df, window=WINDOW, horizon=HORIZON, stride=STRIDE):
    """
    df must already be scaled. Returns:
      X: (n_samples, window, n_features)      -- past readings
      y: (n_samples, horizon)                  -- future soil_moisture
    """
    features = df[FEATURE_COLUMNS].values
    target = df[TARGET_COLUMN].values

    X, y = [], []
    last_start = len(df) - window - horizon
    for start in range(0, last_start + 1, stride):
        X.append(features[start : start + window])
        y.append(target[start + window : start + window + horizon])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_datasets():
    """End-to-end: load -> chronological split -> fit scalers on train ->
    scale all three splits -> window each split independently."""
    df = load_data()
    train_df, val_df, test_df = chronological_split(df)
    scalers = fit_scalers(train_df)

    train_scaled = apply_scalers(train_df, scalers)
    val_scaled = apply_scalers(val_df, scalers)
    test_scaled = apply_scalers(test_df, scalers)

    X_train, y_train = make_windows(train_scaled)
    X_val, y_val = make_windows(val_scaled)
    X_test, y_test = make_windows(test_scaled)

    return {
        "scalers": scalers,
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "raw_splits": (train_df, val_df, test_df),
    }


if __name__ == "__main__":
    data = build_datasets()

    for name in ("train", "val", "test"):
        X, y = data[name]
        print(f"{name:5s} X: {X.shape}   y: {y.shape}")

    X_train, y_train = data["train"]
    print("\nOne example window (scaled features, first 5 of 96 steps):")
    print(X_train[0][:5])
    print("\nCorresponding target (scaled soil_moisture, next 24 steps):")
    print(y_train[0])
