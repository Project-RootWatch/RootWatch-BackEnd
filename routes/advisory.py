from flask import Blueprint, jsonify

from gemini_client import generate_advisory_text
from models import SensorReading

advisory_bp = Blueprint("advisory", __name__, url_prefix="/api/advisory")


def classify_status(reading: SensorReading) -> dict:
    """Placeholder threshold-based status, until the LSTM model is integrated."""
    if reading.soil_moisture < 30:
        return {"level": "warning", "label": "Dry - irrigation likely needed"}
    if reading.soil_moisture > 75:
        return {"level": "serious", "label": "Overwatered - risk of root rot"}
    return {"level": "good", "label": "Normal"}


@advisory_bp.post("")
def get_advisory():
    reading = SensorReading.query.order_by(SensorReading.timestamp.desc()).first()
    if reading is None:
        return jsonify({"error": "No sensor readings yet"}), 404

    status = classify_status(reading)

    try:
        advisory = generate_advisory_text(reading.to_dict(), status["label"])
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Gemini request failed: {e}"}), 502

    return jsonify(
        {
            "status": status,
            "advisory": {"sinhala": advisory.sinhala, "english": advisory.english},
            "based_on_reading": reading.to_dict(),
        }
    )
