"""
Inference for RetinaMNIST — Diabetic Retinopathy grading (5 classes).

Loads the best single-seed model (retinamnist_best.pth) and applies
prior correction (alpha * log_prior subtracted from logits) to
counteract the model's training-frequency bias toward class 0.

Usage:
    from src.inference.predict_retinamnist import predict_retina
    result = predict_retina(image_tensor)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from src.models.cnn_model import CNN
from src.inference.labels import RETINA_LABELS

MODEL_PATH      = "models/retinamnist_best.pth"
LOG_PRIOR_PATH  = "models/retinamnist_log_prior.npy"
ALPHA_PATH      = "models/retinamnist_prior_alpha.npy"
NUM_CLASSES     = 5
IN_CHANNELS     = 3

_model      = None
_log_prior  = None
_alpha      = 0.0
_device     = None


def _load_model(device):
    global _model, _log_prior, _alpha, _device
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_retinamnist.py first."
        )

    _device = device
    _model  = CNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _model.eval()

    # Load prior correction parameters if available
    if os.path.exists(LOG_PRIOR_PATH) and os.path.exists(ALPHA_PATH):
        _log_prior = torch.tensor(
            np.load(LOG_PRIOR_PATH), dtype=torch.float32
        ).to(device)
        _alpha = float(np.load(ALPHA_PATH)[0])
    else:
        _log_prior = None
        _alpha     = 0.0


def predict_retina(image_tensor: torch.Tensor,
                   device: torch.device = None) -> dict:
    """
    Predict diabetic retinopathy grade from a single image tensor.

    Args:
        image_tensor: torch.Tensor shape (3, H, W) or (1, 3, H, W),
                      normalised using retinamnist_stats.json mean/std.
        device: torch.device. Defaults to auto-detect.

    Returns:
        dict:
            "class_id"     : int   — predicted class (0–4)
            "class_name"   : str   — e.g. "Moderate DR"
            "confidence"   : float — softmax probability for top class
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
        # TTA: average with horizontal flip
        logits = (logits + _model(torch.flip(image_tensor, dims=[3]))) / 2.0
        # Prior correction: counteracts training-frequency bias
        if _log_prior is not None and _alpha > 0:
            logits = logits - _alpha * _log_prior
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_id = int(np.argmax(probs))

    return {
        "class_id":      class_id,
        "class_name":    RETINA_LABELS[class_id],
        "confidence":    float(probs[class_id]),
        "probabilities": {RETINA_LABELS[i]: float(probs[i])
                          for i in range(NUM_CLASSES)},
    }