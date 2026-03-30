import os
import numpy as np
from medmnist.dataset import RetinaMNIST


def load_retinamnist(split):
    dataset = RetinaMNIST(
        split=split,
        download=True,
        as_rgb=True
    )
    return dataset


def preprocess_and_save(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    stats = {}

    for split in ["train", "val", "test"]:
        dataset = load_retinamnist(split)

        images = (dataset.imgs / 255.0).astype(np.float32)
        labels = dataset.labels.squeeze().astype(np.int64)

        np.save(f"{output_dir}/retinamnist_{split}_images.npy", images)
        np.save(f"{output_dir}/retinamnist_{split}_labels.npy", labels)

        if split == "train":
            mean = images.mean(axis=(0,1,2))
            std = images.std(axis=(0,1,2))

            stats["train"] = {
                "mean": mean.tolist(),
                "std": std.tolist()
            }

        print(f"{split} images: {images.shape}")
        print(f"{split} labels: {labels.shape}")

    import json

    with open(f"{output_dir}/retinamnist_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\nDataset statistics:")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    preprocess_and_save()