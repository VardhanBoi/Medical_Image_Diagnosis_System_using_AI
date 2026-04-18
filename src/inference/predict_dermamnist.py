"""
Inference for DermaMNIST — 7-class skin lesion classification.

Usage:
    from src.inference.predict_dermamnist import predict_derma
    result = predict_derma(image_tensor)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from src.models.cnn_model import CNN
from src.inference.labels import DERMA_LABELS

MODEL_PATH  = "models/dermamnist_resnet.pth"
NUM_CLASSES = 7
IN_CHANNELS = 3

_model  = None
_device = None


def _load_model(device):
    global _model, _device
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_dermamnist.py first."
        )

    _device = device
    _model  = CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _model.eval()


def predict_derma(image_tensor: torch.Tensor,
                  device: torch.device = None) -> dict:
    """
    Predict skin lesion type from a single image tensor.

    Args:
        image_tensor: torch.Tensor shape (3, H, W) or (1, 3, H, W),
                      normalised with ImageNet mean/std
                      (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225).
        device: torch.device. Defaults to auto-detect.

    Returns:
        dict:
            "class_id"     : int   — predicted class (0–6)
            "class_name"   : str   — e.g. "Melanoma"
            "confidence"   : float — softmax probability
            "probabilities": dict  — {class_name: probability}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _load_model(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = _model(image_tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_id = int(np.argmax(probs))

    return {
        "class_id":      class_id,
        "class_name":    DERMA_LABELS[class_id],
        "confidence":    float(probs[class_id]),
        "probabilities": {DERMA_LABELS[i]: float(probs[i])
                          for i in range(NUM_CLASSES)},
    }