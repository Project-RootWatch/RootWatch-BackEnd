import re
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import create_access_token, create_refresh_token, get_jwt_identity, jwt_required

from extensions import db, limiter
from mailer import send_reset_email
from models import PasswordResetToken, User

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _issue_tokens(user):
    identity = str(user.id)
    return {
        "access_token": create_access_token(identity=identity),
        "refresh_token": create_refresh_token(identity=identity),
    }


@auth_bp.post("/signup")
@limiter.limit("3 per hour")
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Enter a valid email address"}), 400
    if len(password) < MIN_PASSWORD_LENGTH:
        return jsonify({"error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters"}), 400
    if User.query.filter_by(email=email).first() is not None:
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({**_issue_tokens(user), "user": user.to_dict()}), 201


@auth_bp.post("/login")
@limiter.limit("5 per minute")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({**_issue_tokens(user), "user": user.to_dict()})


@auth_bp.get("/me")
@jwt_required()
def me():
    user = db.session.get(User, int(get_jwt_identity()))
    if user is None:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())


@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    # refresh=True means this ONLY accepts a refresh token, never a regular
    # access token — otherwise a leaked access token could be used to mint
    # itself an endless chain of new ones, defeating the point of it being
    # short-lived in the first place.
    identity = get_jwt_identity()
    return jsonify({"access_token": create_access_token(identity=identity)})


@auth_bp.post("/forgot-password")
@limiter.limit("3 per hour")
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    user = User.query.filter_by(email=email).first()
    if user is not None:
        reset = PasswordResetToken.create_for(user)
        db.session.add(reset)
        db.session.commit()

        reset_link = f"{current_app.config['FRONTEND_URL']}/reset-password?token={reset.token}"
        send_reset_email(user.email, reset_link)

    # Same response whether or not the email exists — otherwise this
    # endpoint becomes a way for anyone to check which emails are
    # registered, one guess at a time.
    return jsonify({"message": "If an account exists for that email, a reset link has been sent."})


@auth_bp.post("/reset-password")
@limiter.limit("10 per hour")
def reset_password():
    data = request.get_json(silent=True) or {}
    token = data.get("token") or ""
    password = data.get("password") or ""

    if len(password) < MIN_PASSWORD_LENGTH:
        return jsonify({"error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters"}), 400

    reset = PasswordResetToken.query.filter_by(token=token).first()
    if reset is None or not reset.is_valid():
        return jsonify({"error": "This reset link is invalid or has expired"}), 400

    user = db.session.get(User, reset.user_id)
    user.set_password(password)
    reset.used_at = datetime.now(timezone.utc)
    db.session.commit()

    return jsonify({"message": "Password updated. You can now log in."})
