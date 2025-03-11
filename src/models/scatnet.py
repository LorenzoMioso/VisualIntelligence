import torch
import torch.nn as nn
from kymatio.torch import Scattering2D

from src.config import TARGET_IMAGE_SIZE

# This is my model for binary classification on medical images using ScatNet
# In the current implementation the accuracy is around 83%.
# What can be done to improve the accuracy?
# Is correct to use AdaptiveAvgPool2d(1) to reduce the dimensions?
# Keep in mind that:
# I cannot use ccn layers
# I cannot use transfer learning
# Data ugmentation is already applied
# Classes are balanced
# Images are size 768x768


class FeatureClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        return x


class ScatNetImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # feature extraction
        self.scattering = Scattering2D(
            J=3, shape=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE)
        )
        # pool
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # classifier
        # self.classifier = FeatureClassifier(81) # J = 2
        self.classifier = FeatureClassifier(217)  # J = 3

    def forward(self, x):
        # print(f"forward: {x.shape}")
        x = self.scattering(x)
        # print(f"scattering: {x.shape}")
        x = self.global_pool(x)
        # print(f"global_pool: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"flatten: {x.shape}")
        x = self.classifier(x)
        # print(f"classifier: {x.shape}")
        return x
