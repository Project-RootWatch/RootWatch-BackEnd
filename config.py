import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    BASE_DIR = BASE_DIR
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "rootwatch.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # Safety limit: irrigation valve auto-closes after this many seconds
    # even if the ESP32 never receives/acts on a follow-up "stop" command.
    IRRIGATION_MAX_DURATION_SECONDS = 30

    # Signs/verifies login tokens. Must be set in .env — the fallback here
    # is only so the app doesn't crash on a machine that hasn't configured
    # one yet, and is NOT safe to actually run with (anyone who reads the
    # source could forge tokens).
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-insecure-secret-change-me")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)

    # Used to build the password-reset link. Point this at wherever the web
    # dashboard actually runs.
    FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
