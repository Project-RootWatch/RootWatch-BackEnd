import os

from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager

from config import Config
from extensions import db
from routes.activity import activity_bp
from routes.advisory import advisory_bp
from routes.auth import auth_bp
from routes.forecast import forecast_bp
from routes.irrigation import irrigation_bp
from routes.plant_health import plant_health_bp
from routes.sensors import sensors_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    os.makedirs(os.path.join(Config.BASE_DIR, "instance"), exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    jwt = JWTManager(app)

    # Match our {"error": "..."} shape everywhere instead of
    # flask-jwt-extended's default {"msg": "..."}, so the frontend has one
    # error format to handle.
    @jwt.unauthorized_loader
    def missing_token(reason):
        return jsonify({"error": "Login required"}), 401

    @jwt.invalid_token_loader
    def invalid_token(reason):
        return jsonify({"error": "Invalid or expired session, please log in again"}), 401

    @jwt.expired_token_loader
    def expired_token(header, payload):
        return jsonify({"error": "Session expired, please log in again"}), 401

    app.register_blueprint(auth_bp)
    app.register_blueprint(sensors_bp)
    app.register_blueprint(advisory_bp)
    app.register_blueprint(plant_health_bp)
    app.register_blueprint(irrigation_bp)
    app.register_blueprint(activity_bp)
    app.register_blueprint(forecast_bp)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
