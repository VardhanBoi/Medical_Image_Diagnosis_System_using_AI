"""
Flask API backend for Medical Image Diagnosis System.
Serves predictions for RetinaMNIST, ChestMNIST, DermaMNIST.
"""

import os
import io
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── load retina stats ─────────────────────────────────────────────────────────
_STATS_PATH = "data/retinamnist_stats.json"
if os.path.exists(_STATS_PATH):
    with open(_STATS_PATH) as f:
        _rs = json.load(f)
    RETINA_MEAN = _rs.get("mean", [0.5, 0.5, 0.5])
    RETINA_STD  = _rs.get("std",  [0.5, 0.5, 0.5])
else:
    RETINA_MEAN = [0.5, 0.5, 0.5]
    RETINA_STD  = [0.5, 0.5, 0.5]

# ── per-modality transforms ────────────────────────────────────────────────────
TRANSFORMS = {
    "retina": T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize(mean=RETINA_MEAN, std=RETINA_STD),
    ]),
    "chest": T.Compose([
        T.Resize((28, 28)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]),
    "derma": T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]),
}

# ── lazy imports (heavy — only loaded on first request) ────────────────────────
_predict = None

def _get_predict():
    global _predict
    if _predict is None:
        from src.inference.predict import predict
        _predict = predict
    return _predict


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)                           # allow frontend on any origin


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict_endpoint():
    """
    Expects multipart/form-data:
        file     : image file
        modality : "retina" | "chest" | "derma"
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    modality = request.form.get("modality", "").lower().strip()
    if modality not in ("retina", "chest", "derma"):
        return jsonify({"error": f"Invalid modality '{modality}'. "
                                  "Choose: retina, chest, derma"}), 400

    raw = request.files["file"].read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    transform = TRANSFORMS[modality]
    tensor    = transform(img)          # (C, 28, 28)

    try:
        result = _get_predict()(tensor, modality=modality)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


@app.route("/api/modalities", methods=["GET"])
def modalities():
    return jsonify({
        "modalities": [
            {
                "id":          "retina",
                "label":       "RetinaMNIST",
                "description": "Diabetic Retinopathy Grading (5 classes)",
                "type":        "single-label",
                "classes":     ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"],
                "channels":    "RGB"
            },
            {
                "id":          "chest",
                "label":       "ChestMNIST",
                "description": "Chest X-ray Multi-label (14 pathologies)",
                "type":        "multi-label",
                "classes":     ["Atelectasis","Cardiomegaly","Effusion","Infiltration",
                                "Mass","Nodule","Pneumonia","Pneumothorax","Consolidation",
                                "Edema","Emphysema","Fibrosis","Pleural Thickening","Hernia"],
                "channels":    "Grayscale"
            },
            {
                "id":          "derma",
                "label":       "DermaMNIST",
                "description": "Skin Lesion Classification (7 classes)",
                "type":        "single-label",
                "classes":     ["Actinic Keratoses","Basal Cell Carcinoma","Benign Keratosis",
                                "Dermatofibroma","Melanoma","Melanocytic Nevi","Vascular Lesions"],
                "channels":    "RGB"
            },
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)