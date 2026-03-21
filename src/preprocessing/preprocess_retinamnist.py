import os
import numpy as np
from medmnist.dataset import DermaMNIST


def load_dermamnist(split):
    dataset = DermaMNIST(
        split=split,
        download=True,
        as_rgb=True
    )
    return dataset


def preprocess_and_save(output_dir="processed"):
    os.makedirs(output_dir, exist_ok=True)

    stats = {}

    for split in ["train", "val", "test"]:
        dataset = load_dermamnist(split)

        # Direct array access
        images = dataset.imgs.astype(np.float32) / 255.0
        labels = dataset.labels.astype(np.int64)

        # Save
        np.save(f"{output_dir}/dermamnist_{split}_images.npy", images)
        np.save(f"{output_dir}/dermamnist_{split}_labels.npy", labels)

        stats[split] = {
            "mean": float(images.mean()),
            "std": float(images.std())
        }

        print(f"{split} images: {images.shape}")
        print(f"{split} labels: {labels.shape}")

    print("\nDataset statistics:")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    preprocess_and_save()