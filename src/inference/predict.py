"""
Unified inference entry point for the Medical Image Diagnosis System.

Usage:
    from src.inference.predict import predict
    result = predict(image_tensor, modality="retina")
    result = predict(image_tensor, modality="chest")
    result = predict(image_tensor, modality="derma")
"""

import torch
from typing import Literal

from src.inference.predict_retinamnist import predict_retina
from src.inference.predict_chestmnist  import predict_chest
from src.inference.predict_dermamnist  import predict_derma

SUPPORTED = ("retina", "chest", "derma")


def predict(image_tensor: torch.Tensor,
            modality: Literal["retina", "chest", "derma"],
            device: torch.device = None) -> dict:
    """
    Route an image to the correct model and return predictions.

    Args:
        image_tensor: torch.Tensor (C, H, W) or (1, C, H, W).
                      Normalisation per modality:
                        retina — retinamnist_stats.json mean/std, 3-ch
                        chest  — mean=0.5, std=0.5, 1-ch grayscale
                        derma  — ImageNet mean/std, 3-ch RGB
        modality: "retina" | "chest" | "derma"
        device:   torch.device, defaults to auto-detect.

    Returns:
        Modality-specific prediction dict (see individual predict_*.py).
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
        raise ValueError(f"Unknown modality '{modality}'. Choose from {SUPPORTED}")