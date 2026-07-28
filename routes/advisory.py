from flask import Blueprint, jsonify

from gemini_client import generate_advisory_text
from models import SensorReading

advisory_bp = Blueprint("advisory", __name__, url_prefix="/api/advisory")


def classify_status(reading: SensorReading) -> str:
    """Placeholder threshold-based status, until the LSTM model is integrated."""
    if reading.soil_moisture < 30:
        return "dry - irrigation likely needed"
    if reading.soil_moisture > 75:
        return "overwatered - risk of root rot"
    return "normal"


@advisory_bp.post("")
def get_advisory():
    reading = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    if reading is None:
        return jsonify({"error": "No sensor readings yet"}), 404

    status = classify_status(reading)

    try:
        advisory_text = generate_advisory_text(reading.to_dict(), status)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Gemini request failed: {e}"}), 502

    return jsonify(
        {
            "status": status,
            "advisory_text": advisory_text,
            "based_on_reading": reading.to_dict(),
        }
    )
