import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    """ResNet-34 backbone. Best performer on CPU for this dataset."""

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