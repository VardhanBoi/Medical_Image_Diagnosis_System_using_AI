"""
Unified inference entry point for the Medical Image Diagnosis System.

Routes an image to the correct model based on the dataset/modality.
This is what your app.py or API layer should call.

Usage:
    from src.inference.predict import predict
    result = predict(image_tensor, modality="retina")
    result = predict(image_tensor, modality="chest")
    result = predict(image_tensor, modality="derma")

Or use each predictor directly:
    from src.inference.predict_retinamnist import predict_retina
    from src.inference.predict_chestmnist  import predict_chest
    from src.inference.predict_dermamnist  import predict_derma
"""

import torch
from typing import Literal

from src.inference.predict_retinamnist import predict_retina
from src.inference.predict_chestmnist  import predict_chest
from src.inference.predict_dermamnist  import predict_derma


SUPPORTED_MODALITIES = ("retina", "chest", "derma")


def predict(image_tensor: torch.Tensor,
            modality: Literal["retina", "chest", "derma"],
            device: torch.device = None) -> dict:
    """
    Route an image to the correct model and return predictions.

    Args:
        image_tensor: torch.Tensor — shape (C, H, W) or (1, C, H, W).
                      Must be normalised appropriately for the modality:
                        retina → retinamnist_stats.json mean/std, 3-channel
                        chest  → mean=0.5, std=0.5, 1-channel grayscale
                        derma  → ImageNet mean/std (0.485,.456,.406), 3-channel
        modality:     str — one of "retina", "chest", "derma"
        device:       torch.device. Defaults to auto-detect.

    Returns:
        dict — modality-specific prediction result. See individual
               predict_*.py files for full return schema.

    Raises:
        ValueError: if modality is not one of the supported values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modality == "retina":
        return predict_retina(image_tensor, device=device)
    elif modality == "chest":
        return predict_chest(image_tensor, device=device)
    elif modality == "derma":
        return predict_derma(image_tensor, device=device)
    else:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Choose from: {SUPPORTED_MODALITIES}"
        )


def get_supported_modalities() -> tuple:
    """Return tuple of supported modality strings."""
    return SUPPORTED_MODALITIES