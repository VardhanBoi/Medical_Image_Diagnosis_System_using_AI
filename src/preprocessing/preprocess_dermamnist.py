import os
import numpy as np

from medmnist import INFO
from medmnist.dataset import DermaMNIST


def get_dermamnist(split):
    info = INFO["dermamnist"]
    DataClass = DermaMNIST

    dataset = DataClass(
        split=split,
        download=True,
        as_rgb=True   
    )

    return dataset


def preprocess_and_save(output_dir: str = "processed"):
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        dataset = get_dermamnist(split=split)

        images = []
        labels = []

        for img, label in dataset:
            img = np.array(img)                 # PIL → numpy
            img = img.astype(np.float32) / 255.0

            label = np.array(label).astype(np.int64)

            images.append(img)
            labels.append(label)

        images = np.stack(images)
        labels = np.stack(labels)

        np.save(os.path.join(output_dir, f"dermamnist_{split}_images.npy"), images)
        np.save(os.path.join(output_dir, f"dermamnist_{split}_labels.npy"), labels)

        print(
            f"Saved DermaMNIST {split}: images shape {images.shape}, labels shape {labels.shape}"
        )


if __name__ == "__main__":
    preprocess_and_save()