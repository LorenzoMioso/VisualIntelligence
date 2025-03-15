import torch
import torch.nn as nn


from src.models.classifier import FeatureClassifier


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 7, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class CNNImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 11, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv2 = ConvBlock(8, 16)
        # pool
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # classifier
        self.classifier = FeatureClassifier(16)

    def forward(self, x):
        # print(f"forward: {x.shape}")
        x = self.conv1(x)
        # print(f"conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"conv2: {x.shape}")
        x = self.avgpool(x)
        # print(f"avgpool: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"flatten: {x.shape}")
        x = self.classifier(x)
        return x
