from flask import Blueprint, jsonify, request

from models import IrrigationCommand, PlantScan, SensorReading, to_utc_iso

activity_bp = Blueprint("activity", __name__, url_prefix="/api/activity")


@activity_bp.get("")
def get_activity():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 100))

    events = []

    for r in SensorReading.query.order_by(SensorReading.timestamp.desc()).limit(limit).all():
        events.append(
            {
                "type": "reading",
                "label": f"Moisture {r.soil_moisture:.0f}% recorded",
                "timestamp": to_utc_iso(r.timestamp),
            }
        )

    for c in IrrigationCommand.query.order_by(IrrigationCommand.requested_at.desc()).limit(limit).all():
        events.append(
            {
                "type": "irrigation",
                "label": f"Irrigation triggered - {c.duration_seconds}s",
                "timestamp": to_utc_iso(c.requested_at),
            }
        )

    for s in PlantScan.query.order_by(PlantScan.created_at.desc()).limit(limit).all():
        events.append(
            {
                "type": "plant_scan",
                "label": f"Plant scan - {s.headline}",
                "timestamp": to_utc_iso(s.created_at),
            }
        )

    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return jsonify(events[:limit])
