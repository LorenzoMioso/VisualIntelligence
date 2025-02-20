import json

import pandas as pd
import torch
import torch.nn as nn
from torchinfo import summary

from src.config import TARGET_IMAGE_SIZE, basedir, device
from src.data.dataset import create_dataloaders, create_dataset_df


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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)
        return out


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 11, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.layer1 = ConvBlock(8, 16)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def test_model():
    df = create_dataset_df()
    with open(basedir + "fold_stats.json", "r") as f:
        fold_stats = json.load(f)
    print(fold_stats)
    train_splits = pd.read_csv(basedir + "train_splits.csv")
    val_splits = pd.read_csv(basedir + "val_splits.csv")
    train_idx = train_splits["train_0"].values
    val_idx = val_splits["val_0"].values
    mean = fold_stats["0"]["mean"]
    std = fold_stats["0"]["std"]

    # Create dataloaders with fold-specific normalization
    train_loader, val_loader = create_dataloaders(df, train_idx, val_idx, mean, std)

    # check len of train and val loaders
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    model = ImageClassifier()
    model.to(device)
    images, labels = next(iter(train_loader))
    print(f"Labels : {labels}")

    print(f"Images shape: {images.shape}")
    image = images[0].unsqueeze(0).to(device)
    label = labels[0].unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.inference_mode():
        output = model(image.float())
        print(output)

    # check the model summary
    print(
        summary(
            model,
            input_size=(1, 1, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
            device=device,
        )
    )
