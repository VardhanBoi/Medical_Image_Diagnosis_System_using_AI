import numpy as np
import torch
from torch.utils.data import Dataset


class MedMNISTDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        images = np.load(images_path)
        labels = np.load(labels_path)

        # -------------------------
        # FIX IMAGE SHAPE (ROBUST)
        # -------------------------

        if images.ndim == 3:
            # (N, H, W) → (N, 1, H, W)
            images = np.expand_dims(images, axis=1)

        elif images.ndim == 4:

            if images.shape[1] == 1 or images.shape[1] == 3:
                # Already (N, C, H, W)
                pass

            elif images.shape[-1] == 1 or images.shape[-1] == 3:
                # (N, H, W, C) → (N, C, H, W)
                images = images.transpose(0, 3, 1, 2)

            elif images.shape[2] == 1:
                # (N, H, 1, W) → (N, 1, H, W)
                images = images.transpose(0, 2, 1, 3)

            else:
                raise ValueError(f"Unknown image shape: {images.shape}")

        # Final safety check
        assert images.shape[1] in [1, 3], f"Invalid image shape: {images.shape}"

        # -------------------------
        # FIX LABELS
        # -------------------------

        if labels.ndim == 2 and labels.shape[1] > 1:
            # Multi-label (ChestMNIST)
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            # Single-label (DermaMNIST, RetinaMNIST)
            self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

        self.images = torch.tensor(images, dtype=torch.float32)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------
# GENERIC LOADER
# -------------------------

def get_dataset(dataset_name, split, transform=None):

    images_path = f"data/{dataset_name}_{split}_images.npy"
    labels_path = f"data/{dataset_name}_{split}_labels.npy"

    return MedMNISTDataset(images_path, labels_path, transform)


# -------------------------
# DATASET GETTERS
# -------------------------

def get_dermamnist(split, transform=None):
    return get_dataset("dermamnist", split, transform)


def get_chestmnist(split, transform=None):
    return get_dataset("chestmnist", split, transform)


def get_retinamnist(split, transform=None):
    return get_dataset("retinamnist", split, transform)