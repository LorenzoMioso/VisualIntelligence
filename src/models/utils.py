import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torchinfo import summary

from src.config import TARGET_IMAGE_SIZE, device


class ModelAnalyzer:
    def __init__(self, model, train_loader, val_loader=None):
        """
        Initialize the ModelAnalyzer with common parameters.

        Args:
            model: The PyTorch model to analyze
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def inspect_model_architecture(self):
        """Inspect and print model architecture details"""
        # check len of train and val loaders
        print(f"Train loader length: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val loader length: {len(self.val_loader)}")

        images, labels = next(iter(self.train_loader))
        print(f"Labels : {labels}")

        image = images[0].unsqueeze(0).to(device)
        label = labels[0].unsqueeze(0).to(device)

        # Forward pass
        self.model.eval()
        with torch.inference_mode():
            output = self.model(image.float())
            print(output)

        # check the model summary
        summary(
            self.model,
            input_size=(1, 1, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE),
            device=device,
        )

    def profile_model(self, max_steps=40):
        """
        Profile the model execution

        Args:
            max_steps: Maximum number of profiling steps
        """
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profile"),
        ) as prof:
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                output = self.model(images.float())
                loss = F.cross_entropy(output, labels)
                loss.backward()
                prof.step()
                if prof.step_num == max_steps:
                    break

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("Profiling finished.")
