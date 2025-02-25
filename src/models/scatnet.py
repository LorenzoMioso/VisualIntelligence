import torch
import torch.nn as nn
from kymatio.torch import Scattering2D

from src.config import TARGET_IMAGE_SIZE


# Create a ScatNet
class ScatNetImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction
        self.scattering = Scattering2D(
            J=2, shape=(TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), backend="torch_skcuda"
        )
        # pool
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1 * 81, 2),  # J = 2
            # nn.Linear(3 * 1401, 2), # J = 7
        )

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
