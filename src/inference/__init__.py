"""
src/inference — inference module for the Medical Image Diagnosis System.

Public API:
    predict(image_tensor, modality)  — unified entry point
    predict_retina(image_tensor)     — DR grading, 5 classes
    predict_chest(image_tensor)      — chest X-ray, 14-label multi-label
    predict_derma(image_tensor)      — skin lesion, 7 classes

Label maps:
    RETINA_LABELS, CHEST_LABELS, DERMA_LABELS

Model files required in models/:
    retinamnist_best.pth
    retinamnist_log_prior.npy      (prior correction, from training)
    retinamnist_prior_alpha.npy    (prior correction alpha)
    chestmnist_resnet.pth
    chestmnist_thresholds.npy      (per-class thresholds, from training)
    dermamnist_resnet.pth
"""

from src.inference.predict            import predict
from src.inference.predict_retinamnist import predict_retina
from src.inference.predict_chestmnist  import predict_chest
from src.inference.predict_dermamnist  import predict_derma
from src.inference.labels              import RETINA_LABELS, CHEST_LABELS, DERMA_LABELS

__all__ = [
    "predict",
    "predict_retina",
    "predict_chest",
    "predict_derma",
    "RETINA_LABELS",
    "CHEST_LABELS",
    "DERMA_LABELS",
]