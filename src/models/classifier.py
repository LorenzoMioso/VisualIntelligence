import torch
import torch.nn as nn


class FeatureClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, 16)
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        return x
