from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from extensions import db
from models import IrrigationCommand

irrigation_bp = Blueprint("irrigation", __name__, url_prefix="/api/irrigation")


@irrigation_bp.post("/trigger")
def trigger_irrigation():
    data = request.get_json(silent=True) or {}
    max_duration = current_app.config["IRRIGATION_MAX_DURATION_SECONDS"]

    requested_duration = data.get("duration_seconds", max_duration)
    try:
        requested_duration = int(requested_duration)
    except (TypeError, ValueError):
        return jsonify({"error": "duration_seconds must be an integer"}), 400

    duration_seconds = max(1, min(requested_duration, max_duration))

    command = IrrigationCommand(duration_seconds=duration_seconds)
    db.session.add(command)
    db.session.commit()

    return jsonify(command.to_dict()), 201


@irrigation_bp.get("/command")
def get_pending_command():
    """Polled by the ESP32. Returns the oldest un-picked-up command, if any."""
    command = (
        IrrigationCommand.query.filter(IrrigationCommand.consumed_at.is_(None))
        .order_by(IrrigationCommand.requested_at.asc())
        .first()
    )

    if command is None:
        return jsonify({"action": "none"})

    command.consumed_at = datetime.now(timezone.utc)
    db.session.commit()

    return jsonify({"action": "open", "duration_seconds": command.duration_seconds})


@irrigation_bp.get("/status")
def get_status():
    command = IrrigationCommand.query.order_by(IrrigationCommand.requested_at.desc()).first()
    if command is None:
        return jsonify({"last_command": None})

    return jsonify({"last_command": command.to_dict()})
