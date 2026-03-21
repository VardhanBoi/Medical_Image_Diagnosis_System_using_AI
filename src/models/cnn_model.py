import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.model = models.resnet18(weights="IMAGENET1K_V1")

        if in_channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)