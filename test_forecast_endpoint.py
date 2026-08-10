"""
Ad-hoc verification script for GET /api/forecast — not part of the app,
just exercises it directly via Flask's test client + SQLAlchemy so we can
manipulate the DB precisely without polluting the real dataset.

Run with: ./.venv/Scripts/python.exe test_forecast_endpoint.py
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from flask_jwt_extended import create_access_token

from app import app
from extensions import db
from models import SensorReading, User

SYNTHETIC_CSV = "ml/data/synthetic_readings.csv"


def get_token():
    user = User.query.first()
    if user is None:
        raise SystemExit("No user in the DB — sign up an account first.")
    with app.app_context():
        return create_access_token(identity=str(user.id))


def insert_readings(rows):
    """rows: list of (timestamp, soil_moisture, temperature, light_level)"""
    for ts, moisture, temp, light in rows:
        db.session.add(SensorReading(timestamp=ts, soil_moisture=moisture, temperature=temp, light_level=light))
    db.session.commit()


def delete_readings_after(marker_id):
    SensorReading.query.filter(SensorReading.id > marker_id).delete()
    db.session.commit()


def run():
    with app.app_context():
        token = create_access_token(identity=str(User.query.first().id))
        client = app.test_client()
        headers = {"Authorization": f"Bearer {token}"}

        marker_id = db.session.query(db.func.max(SensorReading.id)).scalar() or 0

        # --- Test A: current real data (only 49 rows) -> expect 409, not enough history ---
        resp = client.get("/api/forecast", headers=headers)
        print("TEST A (real data, insufficient history)")
        print("  status:", resp.status_code)
        print("  body:  ", resp.get_json())
        assert resp.status_code == 409, "expected 409 (not enough history)"
        print("  PASS\n")

        # --- Test B: 96 rows spaced 60s apart (matches real firmware cadence,
        #     mismatches the model's 15-min training assumption) -> expect 409 ---
        now = datetime.now(timezone.utc)
        wrong_cadence_rows = [
            (now - timedelta(seconds=60 * (95 - i)), 50.0, 25.0, 50.0) for i in range(96)
        ]
        insert_readings(wrong_cadence_rows)

        resp = client.get("/api/forecast", headers=headers)
        print("TEST B (96 rows, wrong cadence — 60s instead of 15min)")
        print("  status:", resp.status_code)
        print("  body:  ", resp.get_json())
        assert resp.status_code == 409, "expected 409 (spacing mismatch)"
        print("  PASS\n")

        delete_readings_after(marker_id)

        # --- Test C: 96 rows from the tail of the synthetic CSV, correctly
        #     spaced 15 minutes apart -> expect a real 200 forecast ---
        df = pd.read_csv(SYNTHETIC_CSV, parse_dates=["timestamp"]).tail(96)
        # Re-anchor timestamps to "now" so the spacing check (which only cares
        # about relative gaps, not absolute recency) sees a clean 15-min series.
        base = datetime.now(timezone.utc) - timedelta(minutes=15 * 96)
        synthetic_rows = [
            (base + timedelta(minutes=15 * i), row.soil_moisture, row.temperature, row.light_level)
            for i, row in enumerate(df.itertuples())
        ]
        insert_readings(synthetic_rows)

        resp = client.get("/api/forecast", headers=headers)
        print("TEST C (96 correctly-spaced synthetic rows -> real forecast)")
        print("  status:", resp.status_code)
        body = resp.get_json()
        assert resp.status_code == 200, "expected 200 with a real forecast"
        forecast = body["forecast"]
        assert len(forecast) == 24, "expected 24 forecast points (6h horizon)"
        last_known = synthetic_rows[-1][1]
        print(f"  last known soil_moisture: {last_known:.1f}%")
        print(f"  forecast (24 steps, 15min apart):")
        for point in forecast:
            print(f"    {point['timestamp']}  {point['soil_moisture']:.1f}%")
        print("  PASS\n")

        delete_readings_after(marker_id)

        print("All tests passed. DB restored to original state.")


if __name__ == "__main__":
    run()
