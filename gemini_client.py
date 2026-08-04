from google import genai
from google.genai import types
from pydantic import BaseModel

from config import Config

TEXT_MODEL = "gemini-flash-latest"
VISION_MODEL = "gemini-flash-latest"

_client = None


def get_client():
    global _client
    if _client is None:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


class AdvisoryText(BaseModel):
    sinhala: str
    english: str


class PlantAssessment(BaseModel):
    headline: str
    description: str
    disease: str  # one of: none, mild, severe
    stress: str  # one of: none, mild, moderate, severe
    growth: str  # one of: poor, normal, vigorous


def generate_advisory_text(reading: dict, status: str) -> AdvisoryText:
    color = reading.get("color")
    color_line = (
        f"Leaf color reading (RGB): R={color['r']} G={color['g']} B={color['b']}"
        if color
        else "Leaf color reading: not available (color sensor not connected)"
    )

    prompt = f"""You are an agricultural advisor helping a small-scale farmer in Sri Lanka.
Based on the sensor readings below, write a short, practical advisory (2-4
sentences, plain language, no technical jargon). Be specific about any
action the farmer should take right now, if any. Provide the same advisory
in both Sinhala and English.

Soil moisture: {reading['soil_moisture']}%
Temperature: {reading['temperature']} C
Light level: {reading['light_level']}%
{color_line}
Overall status: {status}
"""
    client = get_client()
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AdvisoryText,
        ),
    )
    return response.parsed


def analyze_plant_photo(image_bytes: bytes, mime_type: str) -> PlantAssessment:
    prompt = """You are an agricultural advisor helping a small-scale farmer in Sri Lanka
assess the health of their plant from a leaf photo. Look at the leaf color,
spots, wilting, or discoloration.

Return:
- headline: a short Sinhala headline (e.g. "රෝගී ලක්ෂණ නොමැත" for "no disease
  detected")
- description: 2-4 sentence Sinhala assessment, plain language, no jargon,
  including any action to take
- disease: exactly one of "none", "mild", "severe"
- stress: exactly one of "none", "mild", "moderate", "severe"
- growth: exactly one of "poor", "normal", "vigorous"
"""
    client = get_client()
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlantAssessment,
        ),
    )
    return response.parsed
