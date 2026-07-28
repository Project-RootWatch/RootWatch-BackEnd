from google import genai

from config import Config

TEXT_MODEL = "gemini-2.5-flash"

_client = None


def get_client():
    global _client
    if _client is None:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


def generate_advisory_text(reading: dict, status: str) -> str:
    prompt = f"""You are an agricultural advisor helping a small-scale farmer in Sri Lanka.
Based on the sensor readings below, write a short, practical advisory in
Sinhala (2-4 sentences, plain language, no technical jargon). Be specific
about any action the farmer should take right now, if any.

Soil moisture: {reading['soil_moisture']}%
Temperature: {reading['temperature']} C
Light level: {reading['light_level']}%
Leaf color reading (RGB): R={reading['color']['r']} G={reading['color']['g']} B={reading['color']['b']}
Overall status: {status}
"""
    client = get_client()
    response = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    return response.text
