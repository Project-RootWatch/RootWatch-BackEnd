"""
Packages everything the backend will need to actually run this model:
  - the trained weights (already at checkpoints/best_model.pt)
  - the exact scaler parameters used at training time, as plain JSON
    (not a pickled sklearn object — avoids sklearn-version pickle
    compatibility issues, and the backend doesn't need scikit-learn as a
    dependency just to apply `(x - mean) / scale`)
  - the architecture hyperparameters needed to reconstruct MoistureLSTM
    before loading the state_dict (a .pt file only stores weight VALUES,
    not the architecture that produced them)

Together, model_metadata.json + best_model.pt are the full contract the
backend integration step needs to honor.
"""

import json
from pathlib import Path

from dataset import build_datasets, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW, HORIZON
from model import MoistureLSTM

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def export_metadata():
    data = build_datasets()
    scalers = data["scalers"]

    scaler_params = {
        col: {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0])}
        for col, scaler in scalers.items()
    }

    model = MoistureLSTM()

    metadata = {
        "feature_columns": FEATURE_COLUMNS,   # order matters — must match inference input order
        "target_column": TARGET_COLUMN,
        "window": WINDOW,                     # steps of history the model expects as input
        "horizon": HORIZON,                   # steps into the future it predicts
        "step_minutes": 15,                   # what one "step" represents in real time
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "scalers": scaler_params,             # apply BEFORE inference, invert AFTER
    }

    out_path = CHECKPOINT_DIR / "model_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2))
    return out_path, metadata


if __name__ == "__main__":
    weights_path = CHECKPOINT_DIR / "best_model.pt"
    if not weights_path.exists():
        raise SystemExit(f"No trained weights found at {weights_path} — run train.py first.")

    out_path, metadata = export_metadata()

    print("Exported model artifacts:")
    print(f"  weights:  {weights_path}")
    print(f"  metadata: {out_path}\n")
    print(json.dumps(metadata, indent=2))
