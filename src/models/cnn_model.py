import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    """
    ResNet-34 backbone.

    Used across all three datasets (RetinaMNIST, ChestMNIST, DermaMNIST).
    ResNet-34 outperforms ResNet-18 on all three:
    - RetinaMNIST: ordinal 5-class grading benefits from deeper features
    - ChestMNIST:  14-label multi-label, larger dataset, deeper is fine
    - DermaMNIST:  texture discrimination between lesion types needs depth

    ResNet-34 was not replaced by ResNet-18 or EfficientNet-B0/B3 because:
    - ResNet-18 has shallower features that conflate adjacent DR grades
    - EfficientNet-B0/B3 underperformed on CPU (dense BN needs large batches)
    - All empirical GPU/CPU runs showed ResNet-34 best on this dataset
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.model       = models.resnet34(weights="IMAGENET1K_V1")

        if in_channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)