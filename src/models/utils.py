import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from torchinfo import summary
from src.models.cnn import CNNImageClassifier
from src.models.scatnet import ScatNetImageClassifier
from src.config import MODEL_CONFIG, device


class ModelAnalyzer:
    """Utility class for analyzing and managing models"""

    def __init__(self, model=None, train_loader=None, val_loader=None):
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
        if self.model is None or self.train_loader is None:
            raise ValueError("Model and train_loader must be provided")

        # Display validation loader info if available
        if self.val_loader:
            print(f"Val loader length: {len(self.val_loader)}")

        # Get a sample batch
        images, labels = next(iter(self.train_loader))
        image = images[0].unsqueeze(0).to(device)
        print(f"Device: {device}")

        # Forward pass with sample
        self.model.eval()
        with torch.inference_mode():
            output = self.model(image.float())
            print(f"Sample output: {output}")

        # Print model summary
        summary(
            self.model,
            input_size=(
                1,
                1,
                MODEL_CONFIG.target_image_size,
                MODEL_CONFIG.target_image_size,
            ),
            device=device,
        )

    def profile_model(self, max_steps=40):
        """
        Profile the model execution
        Args:
            max_steps: Maximum number of profiling steps
        """
        if self.model is None or self.train_loader is None:
            raise ValueError("Model and train_loader must be provided")

        # Configure profiler
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profile"),
        ) as prof:
            # Profile forward pass and backward pass
            for step, (images, labels) in enumerate(self.train_loader):
                if step >= max_steps:
                    break

                images, labels = images.to(device), labels.to(device)
                output = self.model(images.float())
                loss = F.cross_entropy(output, labels)
                loss.backward()
                prof.step()

        # Print profiling results
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("Profiling finished.")

    def load_checkpoint(self, filepath):
        """
        Load a model from a checkpoint file
        Args:
            filepath: Path to the checkpoint file
        Returns:
            The loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=torch.device(device))

        # Get model class name from checkpoint
        model_class_name = checkpoint.get("model_class")
        if not model_class_name:
            raise ValueError("Checkpoint does not contain model_class information")

        model_class = checkpoint["model_class"]
        medelmap = {
            CNNImageClassifier.__name__: CNNImageClassifier,
            ScatNetImageClassifier.__name__: ScatNetImageClassifier,
        }
        model = medelmap[model_class]().to(device)

        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])

        # Prepare for inference
        model.eval()

        # Update internal model reference
        self.model = model
        return model

    def save_checkpoint(
        self, model, filepath, optimizer=None, epoch=None, metrics=None
    ):
        """
        Save a model checkpoint
        Args:
            model: The model to save
            filepath: Path to save the checkpoint
            optimizer: Optional optimizer state
            epoch: Optional current epoch number
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            "model_class": model.__class__.__name__,
            "state_dict": model.state_dict(),
        }

        # Add optional information
        if optimizer:
            checkpoint["optimizer"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, filepath)
