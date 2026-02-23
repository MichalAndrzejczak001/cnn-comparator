import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Pobieramy predefiniowany ResNet-18 bez wstępnie wytrenowanych wag
        self.model = models.resnet18(weights=None)

        # Dostosowujemy pierwszą warstwę konwolucyjną do liczby kanałów wejściowych
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Dostosowujemy warstwę końcową do liczby klas
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
