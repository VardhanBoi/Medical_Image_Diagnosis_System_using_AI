"""
Inference for ChestMNIST — multi-label chest X-ray (14 classes).

ChestMNIST is MULTI-LABEL: one image can have multiple findings.
Uses per-class thresholds tuned on val set during training
(saved to models/chestmnist_thresholds.npy). Falls back to 0.5
if the threshold file is not found.

Usage:
    from src.inference.predict_chestmnist import predict_chest
    result = predict_chest(image_tensor)
"""

import os
import torch
import numpy as np

from src.models.cnn_model import CNN
from src.inference.labels import CHEST_LABELS

MODEL_PATH      = "models/chestmnist_resnet.pth"
THRESHOLD_PATH  = "models/chestmnist_thresholds.npy"
NUM_CLASSES     = 14
IN_CHANNELS     = 1

_model      = None
_thresholds = None
_device     = None


def _load_model(device):
    global _model, _thresholds, _device
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

    # Load per-class thresholds; fall back to 0.5 if missing
    if os.path.exists(THRESHOLD_PATH):
        _thresholds = np.load(THRESHOLD_PATH)
    else:
        _thresholds = np.full(NUM_CLASSES, 0.5)


def predict_chest(image_tensor: torch.Tensor,
                  device: torch.device = None) -> dict:
    """
    Predict chest pathologies from a single chest X-ray tensor.

    Args:
        image_tensor: torch.Tensor shape (1, H, W) or (1, 1, H, W),
                      normalised with mean=0.5, std=0.5.
        device: torch.device. Defaults to auto-detect.

    Returns:
        dict:
            "findings"      : list[str] — detected pathologies
                              (["No Finding"] if none detected)
            "probabilities" : dict — {name: probability} for all 14
            "thresholds"    : dict — {name: threshold used} per class
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _load_model(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = _model(image_tensor).squeeze(0)
        probs  = torch.sigmoid(logits).cpu().numpy()

    preds    = (probs > _thresholds).astype(int)
    findings = [CHEST_LABELS[i] for i in range(NUM_CLASSES) if preds[i] == 1]

    return {
        "findings":      findings if findings else ["No Finding"],
        "probabilities": {CHEST_LABELS[i]: float(probs[i])
                          for i in range(NUM_CLASSES)},
        "thresholds":    {CHEST_LABELS[i]: float(_thresholds[i])
                          for i in range(NUM_CLASSES)},
    }