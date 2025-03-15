import torch
import torch.nn as nn
from kymatio.torch import Scattering2D

from src.config import MODEL_CONFIG

from src.models.classifier import FeatureClassifier


class ScatNetImageClassifier(nn.Module):
    def __init__(self, J=3, L=8, M=2):
        super().__init__()

        # feature extraction
        self.scattering = Scattering2D(
            J=J,
            L=L,
            max_order=M,
            shape=(MODEL_CONFIG.target_image_size, MODEL_CONFIG.target_image_size),
        )
        # pool
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if M == 1:
            feature_size = 1 + L * J
        elif M == 2:
            feature_size = 1 + L * J + (L**2) * J * (J - 1) // 2
        elif M == 3:
            feature_size = (
                1
                + L * J
                + (L**2) * J * (J - 1) // 2
                + (L**3) * J * (J - 1) * (J - 2) // 6
            )

        # classifier
        self.classifier = FeatureClassifier(feature_size)

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
