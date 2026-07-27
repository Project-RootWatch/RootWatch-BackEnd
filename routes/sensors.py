from flask import Blueprint, jsonify, request

from extensions import db
from models import SensorReading

sensors_bp = Blueprint("sensors", __name__, url_prefix="/api/readings")

REQUIRED_FIELDS = ["soil_moisture", "temperature", "light_level", "color_r", "color_g", "color_b"]


@sensors_bp.post("")
def create_reading():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        reading = SensorReading(
            soil_moisture=float(data["soil_moisture"]),
            temperature=float(data["temperature"]),
            light_level=float(data["light_level"]),
            color_r=int(data["color_r"]),
            color_g=int(data["color_g"]),
            color_b=int(data["color_b"]),
        )
    except (TypeError, ValueError):
        return jsonify({"error": "Sensor fields must be numeric"}), 400

    db.session.add(reading)
    db.session.commit()

    return jsonify(reading.to_dict()), 201


@sensors_bp.get("/current")
def get_current_reading():
    reading = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    if reading is None:
        return jsonify({"error": "No readings yet"}), 404

    return jsonify(reading.to_dict())


@sensors_bp.get("/history")
def get_reading_history():
    limit = request.args.get("limit", default=100, type=int)
    limit = max(1, min(limit, 1000))

    readings = (
        SensorReading.query.order_by(SensorReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    readings.reverse()  # oldest first, so charts don't need to re-sort

    return jsonify([r.to_dict() for r in readings])
