import torch
import torch.nn as nn

from src.models.classifier import FeatureClassifier


class CNNImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction con dimensioni ridotte
        self.features = nn.Sequential(
            # Riduzione del numero di filtri iniziali e dimensione kernel
            nn.Conv2d(1, 16, kernel_size=11, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Riduzione dimensione con pooling
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Dimensione finale ridotta: 24x4x4
        )

        # classifier
        self.classifier = FeatureClassifier(24 * 4 * 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
