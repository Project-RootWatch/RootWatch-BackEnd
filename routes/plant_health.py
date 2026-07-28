from flask import Blueprint, jsonify, request

from gemini_client import analyze_plant_photo

plant_health_bp = Blueprint("plant_health", __name__, url_prefix="/api/plant-health")

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_PHOTO_BYTES = 8 * 1024 * 1024  # 8 MB


@plant_health_bp.post("")
def assess_plant_photo():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded. Send as multipart/form-data field 'photo'."}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"error": "Empty file"}), 400

    mime_type = photo.mimetype
    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({"error": f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP."}), 400

    image_bytes = photo.read()
    if len(image_bytes) > MAX_PHOTO_BYTES:
        return jsonify({"error": "Photo too large (max 8MB)"}), 400

    try:
        assessment_text = analyze_plant_photo(image_bytes, mime_type)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Gemini request failed: {e}"}), 502

    return jsonify({"assessment_text": assessment_text})
