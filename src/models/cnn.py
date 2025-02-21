import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from src.config import TARGET_IMAGE_SIZE, device


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


def inspect_model_architecture(model, train_loader, val_loader):
    # check len of train and val loaders
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Labels : {labels}")

    image = images[0].unsqueeze(0).to(device)
    label = labels[0].unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.inference_mode():
        output = model(image.float())
        print(output)

    # check the model summary
    summary(
        model,
        input_size=(1, 1, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
        device=device,
    )


from torch.profiler import ProfilerActivity, profile, record_function


def profile_model(model, train_loader):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile"),
    ) as prof:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images.float())
            loss = F.cross_entropy(output, labels)
            loss.backward()
            prof.step()
            if prof.step_num == 40:
                break

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("Profiling finished.")
