# RootWatch — LSTM soil moisture forecaster

Trains a PyTorch LSTM that reads the last 24 hours of sensor readings
(soil moisture, temperature, light) and predicts the next 6 hours of soil
moisture. Currently trained on **synthetic data** — see below for why, and
what has to happen before this model is trustworthy for real use.

## Why synthetic data

At the time this was built, the real `sensor_readings` table had 49 rows,
45 of them stuck at exactly 100% moisture (an uncalibrated sensor — see
the firmware calibration notes elsewhere in this project), plus a 5.5-day
gap with no data at all. An LSTM trained on that would just learn "always
predict 100%." The synthetic generator produces 4,320 rows (45 days at
15-minute intervals) with a realistic decay/irrigation/day-night pattern
instead, so the pipeline can actually be built and validated end-to-end.

**Before this model is used for real predictions**: fix the sensor
calibration, let the device run continuously for real weeks, export that
history in the same three-column shape (`timestamp, soil_moisture,
temperature, light_level`), and retrain — every script here already
points at a CSV, so swapping the data source is the only change needed.

## Pipeline

| Script | What it does |
|---|---|
| `generate_synthetic_data.py` | Produces `data/synthetic_readings.csv` + a preview plot |
| `dataset.py` | Chronological train/val/test split, per-feature scaling (fit on train only), sliding-window sequence construction |
| `model.py` | `MoistureLSTM` architecture (2-layer LSTM -> linear head, direct multi-step output) |
| `train.py` | Training loop with early stopping on validation loss; saves `checkpoints/best_model.pt` + a loss-curve plot |
| `evaluate.py` | Test-set metrics (MAE/RMSE in real percentage points) vs. a naive "flat" baseline, plus example prediction plots |
| `export.py` | Writes `checkpoints/model_metadata.json` — everything besides the weights that inference needs |

Run them in that order. Each one reads the previous step's output from
`data/` or `checkpoints/`; none of it is wired into the Flask app yet —
that's a deliberately separate next step.

## The inference contract (`checkpoints/`)

Two files together are the full exported artifact:

- **`best_model.pt`** — PyTorch `state_dict` (weights only — you must
  reconstruct `MoistureLSTM` with the matching hyperparameters before
  loading it; a `.pt` file doesn't store architecture).
- **`model_metadata.json`** — everything else inference needs:
  - `feature_columns` — the exact order the model expects input features in
  - `window` / `horizon` / `step_minutes` — 96 steps of 15-minute history in, 24 steps (6h) out
  - `hidden_size` / `num_layers` — needed to reconstruct `MoistureLSTM` before `load_state_dict`
  - `scalers` — per-feature `{mean, scale}`. **Apply `(x - mean) / scale` to every input feature before inference, and invert that same formula on the output** (only `soil_moisture`'s scaler applies to the output, since that's the only thing being predicted). Skipping this step silently produces meaningless predictions — the model has never seen an unscaled number.

## Retraining

```bash
pip install -r requirements.txt
python generate_synthetic_data.py   # or point dataset.py at real data instead
python train.py
python evaluate.py                  # sanity-check before trusting a new checkpoint
python export.py
```
