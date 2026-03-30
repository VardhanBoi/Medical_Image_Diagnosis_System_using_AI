import numpy as np
import torch
from torch.utils.data import Dataset


class MedMNISTDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        images = np.load(images_path)
        labels = np.load(labels_path)

        if images.ndim == 3:
            images = np.expand_dims(images, axis=1)

        elif images.ndim == 4:

            if images.shape[1] == 1 or images.shape[1] == 3:
                pass

            elif images.shape[-1] == 1 or images.shape[-1] == 3:
                images = images.transpose(0, 3, 1, 2)

            elif images.shape[2] == 1:
                images = images.transpose(0, 2, 1, 3)

            else:
                raise ValueError(f"Unknown image shape: {images.shape}")

        assert images.shape[1] in [1, 3], f"Invalid image shape: {images.shape}"

        if labels.ndim == 2 and labels.shape[1] > 1:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long).squeeze()

        self.images = torch.tensor(images, dtype=torch.float32)

        if self.images.max() > 1:
            self.images = self.images / 255.0

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataset(dataset_name, split, transform=None):

    images_path = f"data/{dataset_name}_{split}_images.npy"
    labels_path = f"data/{dataset_name}_{split}_labels.npy"

    return MedMNISTDataset(images_path, labels_path, transform)

def get_dermamnist(split, transform=None):
    return get_dataset("dermamnist", split, transform)


def get_chestmnist(split, transform=None):
    return get_dataset("chestmnist", split, transform)


def get_retinamnist(split, transform=None):
    return get_dataset("retinamnist", split, transform)