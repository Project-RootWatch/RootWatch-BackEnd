from datetime import datetime, timezone

from extensions import db


def to_utc_iso(dt):
    """SQLite drops tzinfo on read, so a value we always wrote as UTC comes
    back naive. Re-attach UTC before formatting or JS Date() misreads it as
    local time on the client."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


class SensorReading(db.Model):
    __tablename__ = "sensor_readings"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    soil_moisture = db.Column(db.Float, nullable=False)  # percentage, 0-100
    temperature = db.Column(db.Float, nullable=False)  # Celsius
    light_level = db.Column(db.Float, nullable=False)  # percentage, 0-100 (from LDR voltage divider)

    # Raw TCS3200 channel readings, used for leaf color tracking over time.
    # Nullable: the TCS3200 isn't wired up on the ESP32 yet, so readings
    # arrive without color data until it is.
    color_r = db.Column(db.Integer, nullable=True)
    color_g = db.Column(db.Integer, nullable=True)
    color_b = db.Column(db.Integer, nullable=True)

    def to_dict(self):
        has_color = self.color_r is not None
        return {
            "id": self.id,
            "timestamp": to_utc_iso(self.timestamp),
            "soil_moisture": self.soil_moisture,
            "temperature": self.temperature,
            "light_level": self.light_level,
            "color": {"r": self.color_r, "g": self.color_g, "b": self.color_b} if has_color else None,
        }


class IrrigationCommand(db.Model):
    __tablename__ = "irrigation_commands"

    id = db.Column(db.Integer, primary_key=True)
    requested_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    duration_seconds = db.Column(db.Integer, nullable=False)

    # Set once the ESP32 has polled and picked up this command
    consumed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "requested_at": to_utc_iso(self.requested_at),
            "duration_seconds": self.duration_seconds,
            "picked_up": self.consumed_at is not None,
            "consumed_at": to_utc_iso(self.consumed_at),
        }


class PlantScan(db.Model):
    __tablename__ = "plant_scans"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    headline = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    disease = db.Column(db.String(20), nullable=False)
    stress = db.Column(db.String(20), nullable=False)
    growth = db.Column(db.String(20), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": to_utc_iso(self.created_at),
            "headline": self.headline,
            "description": self.description,
            "disease": self.disease,
            "stress": self.stress,
            "growth": self.growth,
        }
