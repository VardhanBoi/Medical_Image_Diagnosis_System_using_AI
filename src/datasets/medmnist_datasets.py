import numpy as np
import torch
from torch.utils.data import Dataset


class MedMNISTDataset(Dataset):

    def __init__(self, images_path, labels_path):

        images = np.load(images_path)
        labels = np.load(labels_path)

        # Convert NHWC → NCHW if needed
        if images.ndim == 4:
            images = images.transpose(0,3,1,2)

        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_dataset(dataset_name, split):

    images_path = f"processed/{dataset_name}_{split}_images.npy"
    labels_path = f"processed/{dataset_name}_{split}_labels.npy"

    return MedMNISTDataset(images_path, labels_path)


def get_chestmnist(split):
    return get_dataset("chestmnist", split)


def get_dermamnist(split):
    return get_dataset("dermamnist", split)


def get_retinamnist(split):
    return get_dataset("retinamnist", split)