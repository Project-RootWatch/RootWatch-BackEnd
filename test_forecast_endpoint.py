"""
Ad-hoc verification script for GET /api/forecast — not part of the app,
just exercises it directly via Flask's test client + SQLAlchemy so we can
manipulate the DB precisely without polluting the real dataset.

Run with: ./.venv/Scripts/python.exe test_forecast_endpoint.py
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from flask_jwt_extended import create_access_token

from app import app
from extensions import db
from models import SensorReading, User

SYNTHETIC_CSV = "ml/data/synthetic_readings.csv"
WINDOW = 96
STEP_MINUTES = 15
RAW_INTERVAL_SECONDS = 60  # matches the real firmware's POST_INTERVAL_MS


def insert_readings(rows):
    for ts, moisture, temp, light in rows:
        db.session.add(SensorReading(timestamp=ts, soil_moisture=moisture, temperature=temp, light_level=light))
    db.session.commit()


def delete_readings_after(marker_id):
    SensorReading.query.filter(SensorReading.id > marker_id).delete()
    db.session.commit()


def synthetic_15min_tail(n=WINDOW):
    df = pd.read_csv(SYNTHETIC_CSV, parse_dates=["timestamp"]).tail(n)
    return list(df.itertuples())


def upsample_to_raw_cadence(quarter_hour_points, base_time, rng):
    """Each 15-min synthetic point becomes 15 raw readings 60s apart, with
    small noise added so the backend's averaging is doing real work rather
    than just averaging copies of one number."""
    rows = []
    steps_per_bucket = (STEP_MINUTES * 60) // RAW_INTERVAL_SECONDS
    for i, point in enumerate(quarter_hour_points):
        bucket_start = base_time + timedelta(minutes=STEP_MINUTES * i)
        for j in range(steps_per_bucket):
            ts = bucket_start + timedelta(seconds=RAW_INTERVAL_SECONDS * j)
            moisture = point.soil_moisture + rng.normal(0, 0.3)
            temp = point.temperature + rng.normal(0, 0.1)
            light = np.clip(point.light_level + rng.normal(0, 1), 0, 100)
            rows.append((ts, moisture, temp, light))
    return rows


def run():
    rng = np.random.default_rng(11)

    with app.app_context():
        token = create_access_token(identity=str(User.query.first().id))
        client = app.test_client()
        headers = {"Authorization": f"Bearer {token}"}

        marker_id = db.session.query(db.func.max(SensorReading.id)).scalar() or 0

        # Everything from here down runs inside try/finally — a failed
        # assertion must never skip cleanup and leave test rows sitting in
        # the real database (which is exactly what happened the first time
        # this script was run without this guard).
        try:
            # --- Test A: current real data — sparse, big gaps -> expect 409 ---
            resp = client.get("/api/forecast", headers=headers)
            print("TEST A (real data, sparse/gappy)")
            print("  status:", resp.status_code)
            print("  body:  ", resp.get_json())
            assert resp.status_code == 409
            print("  PASS\n")

            # --- Test B: 24h of raw 60s-cadence data with ONE 15-min bucket
            #     deliberately missing -> expect a clear gap error ---
            points = synthetic_15min_tail()
            base = datetime.now(timezone.utc) - timedelta(minutes=STEP_MINUTES * WINDOW)
            raw_rows = upsample_to_raw_cadence(points, base, rng)

            # The endpoint anchors its bucket grid to the ACTUAL latest raw
            # reading (not our `base` variable) — each bucket only gets 14 of
            # its 15 minutes filled (15 samples * 60s), so the true last raw
            # reading lands 1 minute short of base+24h. Derive the gap window
            # from that same real anchor, or it won't line up with a bucket
            # boundary the endpoint actually uses.
            real_latest = raw_rows[-1][0]
            real_start = real_latest - timedelta(minutes=STEP_MINUTES * WINDOW)
            gap_bucket = 40  # remove all raw rows belonging to this 15-min bucket
            gap_start = real_start + timedelta(minutes=STEP_MINUTES * gap_bucket)
            gap_end = gap_start + timedelta(minutes=STEP_MINUTES)
            raw_rows_with_gap = [r for r in raw_rows if not (gap_start <= r[0] < gap_end)]

            insert_readings(raw_rows_with_gap)
            resp = client.get("/api/forecast", headers=headers)
            print("TEST B (24h of 60s-cadence data, one 15-min bucket missing)")
            print("  status:", resp.status_code)
            print("  body:  ", resp.get_json())
            assert resp.status_code == 409
            print("  PASS\n")

            delete_readings_after(marker_id)

            # --- Test C: full 24h of 60s-cadence data, no gaps -> expect a
            #     real forecast, proving the resampling/averaging works ---
            insert_readings(raw_rows)  # the complete set, no gap removed this time
            resp = client.get("/api/forecast", headers=headers)
            print("TEST C (full 24h of 60s-cadence data -> resampled forecast)")
            print("  status:", resp.status_code)
            body = resp.get_json()
            assert resp.status_code == 200, body
            forecast = body["forecast"]
            assert len(forecast) == 24
            print(f"  last known (raw, noisy) soil_moisture ~ {points[-1].soil_moisture:.1f}%")
            for point in forecast[:5]:
                print(f"    {point['timestamp']}  {point['soil_moisture']:.1f}%")
            print("    ...")
            print("  PASS\n")

            print("All tests passed.")
        finally:
            delete_readings_after(marker_id)
            remaining = SensorReading.query.count()
            print(f"Cleanup done — {remaining} rows remain (should match what existed before this script ran).")


if __name__ == "__main__":
    run()
