import json
from datetime import timedelta
from pathlib import Path

import torch
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

from ml.model import MoistureLSTM
from models import SensorReading, to_utc_iso

forecast_bp = Blueprint("forecast", __name__, url_prefix="/api/forecast")

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "ml" / "checkpoints"
METADATA_PATH = CHECKPOINT_DIR / "model_metadata.json"
WEIGHTS_PATH = CHECKPOINT_DIR / "best_model.pt"

# How far real reading spacing is allowed to drift from the model's
# trained step size before we refuse to predict rather than silently
# feeding mismatched-cadence data through it.
SPACING_TOLERANCE = 0.5  # 50%

_model = None
_metadata = None


def _load_model():
    """Loads once per process and caches — reloading a model from disk on
    every request would be needlessly slow."""
    global _model, _metadata
    if _model is not None:
        return _model, _metadata

    if not METADATA_PATH.exists() or not WEIGHTS_PATH.exists():
        raise RuntimeError("No trained forecast model found. Run the ml/ training pipeline first.")

    metadata = json.loads(METADATA_PATH.read_text())
    model = MoistureLSTM(
        n_features=len(metadata["feature_columns"]),
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        horizon=metadata["horizon"],
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    _model, _metadata = model, metadata
    return _model, _metadata


def _scale(value, params):
    return (value - params["mean"]) / params["scale"]


def _unscale(value, params):
    return value * params["scale"] + params["mean"]


def predict_forecast():
    """Returns (forecast_points, error_message) — exactly one is None."""
    model, metadata = _load_model()
    window = metadata["window"]
    feature_columns = metadata["feature_columns"]
    scalers = metadata["scalers"]
    expected_step = timedelta(minutes=metadata["step_minutes"])

    readings = SensorReading.query.order_by(SensorReading.timestamp.desc()).limit(window).all()
    readings.reverse()  # oldest -> newest, matching training order

    if len(readings) < window:
        return None, (
            f"Not enough history yet: need {window} readings, only have {len(readings)}. "
            "The forecast will become available once enough real data has accumulated."
        )

    # Guard against feeding the model data at the wrong cadence (see
    # module docstring-equivalent note above — the firmware currently
    # posts every 60s, the model expects 15-minute steps).
    gaps = [
        (readings[i].timestamp - readings[i - 1].timestamp).total_seconds() for i in range(1, len(readings))
    ]
    median_gap = sorted(gaps)[len(gaps) // 2]
    expected_seconds = expected_step.total_seconds()
    if abs(median_gap - expected_seconds) > expected_seconds * SPACING_TOLERANCE:
        return None, (
            f"Readings are spaced ~{median_gap:.0f}s apart, but this model was trained on "
            f"{expected_seconds:.0f}s steps. Predicting anyway would silently produce a "
            "meaningless forecast — the firmware's posting interval and the model's training "
            "interval need to match (or real data needs resampling to match) before this works."
        )

    rows = []
    for r in readings:
        row = []
        for col in feature_columns:
            raw = getattr(r, col)
            if raw is None:
                return None, f"Reading at {to_utc_iso(r.timestamp)} is missing '{col}' — forecasting needs complete rows."
            row.append(_scale(raw, scalers[col]))
        rows.append(row)

    x = torch.tensor([rows], dtype=torch.float32)  # (1, window, n_features)
    with torch.no_grad():
        pred_scaled = model(x).numpy()[0]  # (horizon,)

    target_scaler = scalers[metadata["target_column"]]
    predictions = [round(_unscale(float(v), target_scaler), 2) for v in pred_scaled]

    last_timestamp = readings[-1].timestamp
    forecast_points = [
        {
            "timestamp": to_utc_iso(last_timestamp + expected_step * (i + 1)),
            "soil_moisture": p,
        }
        for i, p in enumerate(predictions)
    ]

    return forecast_points, None


@forecast_bp.get("")
@jwt_required()
def get_forecast():
    try:
        points, error = predict_forecast()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    if error:
        return jsonify({"error": error}), 409  # not a bad request — just not ready yet

    return jsonify({"forecast": points})
