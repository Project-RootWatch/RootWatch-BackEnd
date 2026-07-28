import os

from flask import Flask

from config import Config
from extensions import db
from routes.advisory import advisory_bp
from routes.sensors import sensors_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(os.path.join(Config.BASE_DIR, "instance"), exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    app.register_blueprint(sensors_bp)
    app.register_blueprint(advisory_bp)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
