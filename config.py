import os

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
