"""
Inference for RetinaMNIST — Diabetic Retinopathy grading (5 classes).

Uses the multi-seed ensemble saved during training.
Each seed model votes via probability averaging (soft ensemble).

Usage:
    from src.inference.predict_retinamnist import predict_retina
    result = predict_retina(image_tensor)   # (C, H, W) float tensor, normalised
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from src.models.cnn_model import CNN
from src.inference.labels import RETINA_LABELS

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
SEED_MODELS = [
    "retinamnist_seed314_best.pth",
    "retinamnist_seed42_best.pth",
    "retinamnist_seed777_best.pth",
    "retinamnist_seed1337_best.pth",
]
WEIGHTS_FILE = "retinamnist_ensemble_weights.npy"
NUM_CLASSES  = 5
IN_CHANNELS  = 3

_models       = None   # cached after first load
_ens_weights  = None
_device       = None


def _load_models(device):
    global _models, _ens_weights, _device
    if _models is not None:
        return

    _device = device
    weights_path = os.path.join(MODEL_DIR, WEIGHTS_FILE)

    if os.path.exists(weights_path):
        _ens_weights = np.load(weights_path)
    else:
        # Fall back to equal weights if ensemble file not found
        _ens_weights = np.ones(len(SEED_MODELS)) / len(SEED_MODELS)

    _models = []
    for i, fname in enumerate(SEED_MODELS):
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping")
            continue
        m = CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        _models.append((m, _ens_weights[i]))

    if not _models:
        raise FileNotFoundError(
            "No RetinaMNIST model files found in models/. "
            "Run train_retinamnist.py first."
        )


def predict_retina(image_tensor: torch.Tensor,
                   device: torch.device = None) -> dict:
    """
    Predict diabetic retinopathy grade from a single image tensor.

    Args:
        image_tensor: torch.Tensor of shape (3, H, W) or (1, 3, H, W),
                      normalised using retinamnist_stats.json mean/std.
        device: torch.device. Defaults to cpu.

    Returns:
        dict with keys:
            "class_id"     : int — predicted class index (0–4)
            "class_name"   : str — e.g. "Moderate DR"
            "confidence"   : float — ensemble probability for top class
            "probabilities": dict — {class_name: probability} for all classes
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _load_models(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)   # add batch dim
    image_tensor = image_tensor.to(device)

    # Weighted soft ensemble: average probabilities across seed models
    ensemble_probs = torch.zeros(1, NUM_CLASSES, device=device)
    total_weight   = 0.0

    for model, w in _models:
        with torch.no_grad():
            logits = model(image_tensor)
            # TTA: average with horizontal flip
            logits_flip = model(torch.flip(image_tensor, dims=[3]))
            logits = (logits + logits_flip) / 2.0
        probs          = F.softmax(logits, dim=1)
        ensemble_probs += w * probs
        total_weight   += w

    ensemble_probs /= total_weight
    probs_np = ensemble_probs.squeeze(0).cpu().numpy()

    class_id = int(np.argmax(probs_np))

    return {
        "class_id":      class_id,
        "class_name":    RETINA_LABELS[class_id],
        "confidence":    float(probs_np[class_id]),
        "probabilities": {
            RETINA_LABELS[i]: float(probs_np[i])
            for i in range(NUM_CLASSES)
        },
    }