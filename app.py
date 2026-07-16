"""
app.py - Flask Backend
========================
Serves the web UI and exposes prediction + history API endpoints.
"""

import os
import random
import json
import numpy
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from model import load_model
from utils import extract_feature_vector, get_feature_display, extract_features

# ─── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load model once at startup ────────────────────────────────────────────
try:
    MODEL, MODEL_META = load_model()
    print(f"[Flask] Model loaded ✓  Accuracy: {MODEL_META['accuracy']}%")
except FileNotFoundError as e:
    print(f"[Flask] WARNING: {e}")
    MODEL, MODEL_META = None, {}

# ─── In-memory history (last 50 scans) ───────────────────────────────────
scan_history = []
MAX_HISTORY = 50


# ─── Helpers ──────────────────────────────────────────────────────────────

URL_REGEX = re.compile(
    r"^(https?://)?"
    r"(([a-zA-Z0-9\-\.]+)\.[a-zA-Z]{2,})"
    r"(:\d+)?"
    r"(/[^\s]*)?"
    r"(\?[^\s]*)?"
    r"(#[^\s]*)?$"
)

def validate_url(url: str) -> tuple[bool, str]:
    """Basic URL validation before feature extraction."""
    url = url.strip()
    if not url:
        return False, "URL cannot be empty."
    if len(url) > 2048:
        return False, "URL exceeds maximum length (2048 characters)."
    if not URL_REGEX.match(url):
        return False, "Invalid URL format. Please include http:// or https://."
    return True, ""


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def risk_level_from_prob(prob: float) -> str:
    if prob >= 0.80:
        return "critical"
    elif prob >= 0.60:
        return "high"
    elif prob >= 0.40:
        return "medium"
    else:
        return "low"


# ─── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: { "url": "https://example.com" }
    Returns: prediction, confidence, features, metadata
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run `python model.py` first."}), 503

    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Request body must contain a 'url' field."}), 400

    raw_url = data["url"]
    url     = normalize_url(raw_url)

    valid, msg = validate_url(url)
    if not valid:
        return jsonify({"error": msg}), 422

    # ── Feature extraction ──
    try:
        feature_vector  = extract_feature_vector(url)
        feature_display = get_feature_display(url)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

    # ── Prediction ──
    try:
        X         = np.array([feature_vector])
        proba     = MODEL.predict_proba(X)[0]
        pred      = int(MODEL.predict(X)[0])
        phish_prob = float(proba[1])
        legit_prob = float(proba[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    label      = "Phishing" if pred == 1 else "Legitimate"
    confidence = phish_prob if pred == 1 else legit_prob
    risk       = risk_level_from_prob(phish_prob)

    # ── Store in history ──
    record = {
        "url": url,
        "label": label,
        "confidence": round(confidence * 100, 1),
        "phish_prob": round(phish_prob * 100, 1),
        "legit_prob": round(legit_prob * 100, 1),
        "risk": risk,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    scan_history.insert(0, record)
    if len(scan_history) > MAX_HISTORY:
        scan_history.pop()

    return jsonify({
        "url": url,
        "prediction": pred,
        "label": label,
        "confidence": round(confidence * 100, 1),
        "phish_probability": round(phish_prob * 100, 1),
        "legit_probability": round(legit_prob * 100, 1),
        "risk_level": risk,
        "features": feature_display,
        "timestamp": record["timestamp"],
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model accuracy and metadata for the UI dashboard."""
    if not MODEL_META:
        return jsonify({"error": "Model metadata unavailable."}), 503
    return jsonify(MODEL_META)


@app.route("/api/history", methods=["GET"])
def history():
    """Return the last N scan results."""
    limit = min(int(request.args.get("limit", 10)), MAX_HISTORY)
    return jsonify({"history": scan_history[:limit]})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return aggregate statistics from the current session."""
    if not scan_history:
        return jsonify({"total": 0, "phishing": 0, "legitimate": 0, "phish_rate": 0})
    total   = len(scan_history)
    phishing = sum(1 for r in scan_history if r["label"] == "Phishing")
    legit    = total - phishing
    return jsonify({
        "total": total,
        "phishing": phishing,
        "legitimate": legit,
        "phish_rate": round(phishing / total * 100, 1),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


# ─── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  AI-POWERED PHISHING DETECTION SYSTEM")
    print("  http://127.0.0.1:5000")
    print("═" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
