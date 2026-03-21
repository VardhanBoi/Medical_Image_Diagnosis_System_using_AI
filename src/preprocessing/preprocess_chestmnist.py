import os
import numpy as np
from medmnist.dataset import DermaMNIST


def preprocess_and_save(output_dir="processed"):

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train","val","test"]:

        dataset = DermaMNIST(
            split=split,
            download=True,
            as_rgb=True
        )

        images = dataset.imgs.astype(np.float32) / 255.0
        labels = dataset.labels.astype(np.int64)

        np.save(f"{output_dir}/dermamnist_{split}_images.npy", images)
        np.save(f"{output_dir}/dermamnist_{split}_labels.npy", labels)

        print(split, images.shape, labels.shape)


if __name__ == "__main__":
    preprocess_and_save()