from datetime import datetime, timezone

from extensions import db


class SensorReading(db.Model):
    __tablename__ = "sensor_readings"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    soil_moisture = db.Column(db.Float, nullable=False)  # percentage, 0-100
    temperature = db.Column(db.Float, nullable=False)  # Celsius
    light_level = db.Column(db.Float, nullable=False)  # percentage, 0-100 (from LDR voltage divider)

    # Raw TCS3200 channel readings, used for leaf color tracking over time
    color_r = db.Column(db.Integer, nullable=False)
    color_g = db.Column(db.Integer, nullable=False)
    color_b = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "soil_moisture": self.soil_moisture,
            "temperature": self.temperature,
            "light_level": self.light_level,
            "color": {"r": self.color_r, "g": self.color_g, "b": self.color_b},
        }
