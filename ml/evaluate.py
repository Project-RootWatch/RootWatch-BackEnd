"""
Evaluates the best checkpoint on the held-out test set — data the model
never trained on AND never influenced early-stopping decisions.

Reports:
  - MAE/RMSE in real percentage-point units (not scaled units — those
    aren't interpretable to a human)
  - Comparison against a naive "moisture stays flat" baseline, since a
    forecasting model is only actually useful if it beats that
  - A few example plots: history -> actual future vs predicted future
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import build_datasets, HORIZON
from model import MoistureLSTM

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best_model.pt"
DATA_DIR = Path(__file__).parent / "data"


def inverse_scale_moisture(scaled_values, scalers):
    """Undo the StandardScaler transform to get back real 0-100 percentages."""
    scaler = scalers["soil_moisture"]
    flat = scaled_values.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(scaled_values.shape)


def evaluate():
    data = build_datasets()
    X_test, y_test = data["test"]
    scalers = data["scalers"]

    model = MoistureLSTM()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(X_test)).numpy()

    # Back to real percentage-point units for both predictions and targets
    preds = inverse_scale_moisture(preds_scaled, scalers)
    actual = inverse_scale_moisture(y_test, scalers)

    mae = np.mean(np.abs(preds - actual))
    rmse = np.sqrt(np.mean((preds - actual) ** 2))

    # Naive baseline: "moisture stays exactly where it is right now" for
    # the whole horizon. The last step of the (unscaled) input window is
    # the most recent real reading before the forecast starts.
    last_known_scaled = X_test[:, -1, 0]  # soil_moisture is feature index 0
    last_known = inverse_scale_moisture(last_known_scaled, scalers)
    naive_preds = np.repeat(last_known[:, None], HORIZON, axis=1)
    naive_mae = np.mean(np.abs(naive_preds - actual))
    naive_rmse = np.sqrt(np.mean((naive_preds - actual) ** 2))

    print(f"Test set size: {len(X_test)} windows\n")
    print(f"{'':20s}{'MAE (pts)':>12s}{'RMSE (pts)':>12s}")
    print(f"{'LSTM':20s}{mae:12.2f}{rmse:12.2f}")
    print(f"{'naive (flat)':20s}{naive_mae:12.2f}{naive_rmse:12.2f}")

    improvement = (naive_mae - mae) / naive_mae * 100
    print(f"\nLSTM beats the naive baseline by {improvement:.1f}% (lower MAE is better).")

    plot_examples(X_test, actual, preds, scalers)


def plot_examples(X_test, actual, preds, scalers, n_examples=4):
    history_scaled = X_test[:, :, 0]  # soil_moisture across the input window
    history = inverse_scale_moisture(history_scaled, scalers)

    rng = np.random.default_rng(7)
    indices = rng.choice(len(X_test), size=n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 1, figsize=(9, 3 * n_examples), sharex=True)
    window = history.shape[1]
    horizon = actual.shape[1]

    for ax, idx in zip(axes, indices):
        hist_x = np.arange(-window, 0)
        future_x = np.arange(0, horizon)

        ax.plot(hist_x, history[idx], color="#888", label="history (input)")
        ax.plot(future_x, actual[idx], color="#22c55e", label="actual future")
        ax.plot(future_x, preds[idx], color="#f2a93c", linestyle="--", label="predicted future")
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_ylabel("moisture (%)")

    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("steps relative to forecast start (15 min each)")
    fig.suptitle("LSTM forecasts on unseen test windows")
    fig.tight_layout()

    out_path = DATA_DIR / "test_predictions.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved example predictions to {out_path}")


if __name__ == "__main__":
    evaluate()
