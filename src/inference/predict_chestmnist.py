"""
Inference for ChestMNIST — multi-label chest X-ray pathology detection.

ChestMNIST is MULTI-LABEL: a single image can have multiple findings
simultaneously (e.g. Effusion + Atelectasis). Each class is treated as
an independent binary classification with a sigmoid threshold.

Usage:
    from src.inference.predict_chestmnist import predict_chest
    result = predict_chest(image_tensor)   # (1, H, W) grayscale tensor
"""

import os
import torch
import numpy as np

from src.models.cnn_model import CNN
from src.inference.labels import CHEST_LABELS

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH  = "models/chestmnist_resnet.pth"
NUM_CLASSES = 14
IN_CHANNELS = 1
THRESHOLD   = 0.5   # sigmoid threshold for positive prediction

_model  = None
_device = None


def _load_model(device):
    global _model, _device
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_chestmnist.py first."
        )
    _device = device
    _model  = CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _model.eval()


def predict_chest(image_tensor: torch.Tensor,
                  device: torch.device = None,
                  threshold: float = THRESHOLD) -> dict:
    """
    Predict chest pathologies from a single chest X-ray image tensor.

    Args:
        image_tensor: torch.Tensor shape (1, H, W) or (1, 1, H, W),
                      normalised with mean=0.5, std=0.5.
        device:    torch.device. Defaults to cpu.
        threshold: float — sigmoid threshold for positive class (default 0.5).

    Returns:
        dict with keys:
            "findings"      : list[str] — detected pathology names
                              (empty list means "No Finding")
            "probabilities" : dict — {pathology_name: probability} for all 14 classes
            "raw_scores"    : dict — {pathology_name: logit} for threshold tuning
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _load_model(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = _model(image_tensor).squeeze(0)   # (14,)
        probs  = torch.sigmoid(logits).cpu().numpy()

    preds    = (probs > threshold).astype(int)
    findings = [CHEST_LABELS[i] for i in range(NUM_CLASSES) if preds[i] == 1]

    return {
        "findings":      findings if findings else ["No Finding"],
        "probabilities": {CHEST_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "raw_scores":    {CHEST_LABELS[i]: float(logits[i].cpu()) for i in range(NUM_CLASSES)},
    }