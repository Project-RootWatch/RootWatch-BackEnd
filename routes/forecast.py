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


def _resample_to_buckets(readings, window, step, feature_columns):
    """
    The ESP32 posts every ~60s, but the model was trained on 15-minute
    steps — so raw rows can't go straight into the model. Instead we
    bucket the raw readings into `window` consecutive `step`-sized windows
    (anchored to the latest reading, working backwards) and average each
    bucket's values, one averaged point per training step.

    Returns (rows, error_message) — exactly one is None. `rows` is a list
    of `window` [feature...] lists in chronological order, ready to scale.

    A bucket with zero raw readings in it is a real gap in history — we
    refuse to interpolate or fabricate a value for it, since that would
    quietly feed the model a number nothing actually measured.
    """
    if not readings:
        return None, "No sensor readings yet."

    latest = readings[-1].timestamp
    start = latest - step * window

    buckets = [[] for _ in range(window)]
    for r in readings:
        if r.timestamp < start:
            continue
        index = int((r.timestamp - start) / step)
        if 0 <= index < window:
            buckets[index].append(r)

    rows = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            gap_start = to_utc_iso(start + step * i)
            gap_end = to_utc_iso(start + step * (i + 1))
            return None, (
                f"Missing sensor data between {gap_start} and {gap_end} — can't build a full "
                f"{window * step.total_seconds() / 3600:.0f}-hour history without it."
            )
        rows.append([sum(getattr(r, col) for r in bucket) / len(bucket) for col in feature_columns])

    return rows, None


def predict_forecast():
    """Returns (forecast_points, error_message) — exactly one is None."""
    model, metadata = _load_model()
    window = metadata["window"]
    feature_columns = metadata["feature_columns"]
    scalers = metadata["scalers"]
    step = timedelta(minutes=metadata["step_minutes"])

    # Raw readings can arrive every ~60s, so covering `window` training
    # steps needs up to `window * step` worth of raw history, not just
    # `window` raw rows.
    lookback_start_guess = None
    latest = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    if latest is not None:
        lookback_start_guess = latest.timestamp - step * window

    readings = (
        SensorReading.query.filter(SensorReading.timestamp >= lookback_start_guess).order_by(SensorReading.timestamp).all()
        if lookback_start_guess is not None
        else []
    )

    rows, error = _resample_to_buckets(readings, window, step, feature_columns)
    if error:
        return None, error

    scaled_rows = [[_scale(value, scalers[col]) for value, col in zip(row, feature_columns)] for row in rows]

    x = torch.tensor([scaled_rows], dtype=torch.float32)  # (1, window, n_features)
    with torch.no_grad():
        pred_scaled = model(x).numpy()[0]  # (horizon,)

    target_scaler = scalers[metadata["target_column"]]
    predictions = [round(_unscale(float(v), target_scaler), 2) for v in pred_scaled]

    last_timestamp = readings[-1].timestamp
    forecast_points = [
        {
            "timestamp": to_utc_iso(last_timestamp + step * (i + 1)),
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
