"""
src/inference — inference module for Medical Image Diagnosis System.

Public API:
    predict(image_tensor, modality)   — unified entry point
    predict_retina(image_tensor)      — RetinaMNIST (DR grading, 5 classes)
    predict_chest(image_tensor)       — ChestMNIST  (14-class multi-label)
    predict_derma(image_tensor)       — DermaMNIST  (skin lesion, 7 classes)

Label maps:
    RETINA_LABELS, CHEST_LABELS, DERMA_LABELS
"""

from src.inference.predict            import predict, get_supported_modalities
from src.inference.predict_retinamnist import predict_retina
from src.inference.predict_chestmnist  import predict_chest
from src.inference.predict_dermamnist  import predict_derma
from src.inference.labels              import RETINA_LABELS, CHEST_LABELS, DERMA_LABELS

__all__ = [
    "predict",
    "get_supported_modalities",
    "predict_retina",
    "predict_chest",
    "predict_derma",
    "RETINA_LABELS",
    "CHEST_LABELS",
    "DERMA_LABELS",
]