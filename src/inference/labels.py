"""
Human-readable label maps for all three datasets.
Used by all inference scripts to turn class indices into disease names.
"""

RETINA_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

DERMA_LABELS = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevi",
    6: "Vascular Lesions",
}

# ChestMNIST is multi-label — each image can have multiple findings
CHEST_LABELS = {
    0:  "Atelectasis",
    1:  "Cardiomegaly",
    2:  "Effusion",
    3:  "Infiltration",
    4:  "Mass",
    5:  "Nodule",
    6:  "Pneumonia",
    7:  "Pneumothorax",
    8:  "Consolidation",
    9:  "Edema",
    10: "Emphysema",
    11: "Fibrosis",
    12: "Pleural Thickening",
    13: "Hernia",
}