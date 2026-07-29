from flask import Blueprint, jsonify, request

from models import IrrigationCommand, PlantScan, SensorReading

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
                "timestamp": r.timestamp.isoformat(),
            }
        )

    for c in IrrigationCommand.query.order_by(IrrigationCommand.requested_at.desc()).limit(limit).all():
        events.append(
            {
                "type": "irrigation",
                "label": f"Irrigation triggered - {c.duration_seconds}s",
                "timestamp": c.requested_at.isoformat(),
            }
        )

    for s in PlantScan.query.order_by(PlantScan.created_at.desc()).limit(limit).all():
        events.append(
            {
                "type": "plant_scan",
                "label": f"Plant scan - {s.headline}",
                "timestamp": s.created_at.isoformat(),
            }
        )

    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return jsonify(events[:limit])
