import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)

from src.config import device


class XAI_CAPTUM:
    """Class for visualizing neural network model explanations and behavior."""

    def __init__(self, model):
        """
        Initialize the visualizer with a PyTorch model

        Args:
            model: The PyTorch model to visualize
        """
        self.model = model
        self.model.eval()

    def normalize_image_for_display(self, image):
        """
        Normalize image data for display

        Args:
            image_array: The image array to normalize

        Returns:
            Normalized image array
        """
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max()
        return image

    def visualize_model_attributions(self, image, target=None):
        """
        Visualize different attribution methods for the model and input

        Args:
            input_tensor: Input tensor with shape [1, C, H, W]
            target: Target class (optional, will use predicted class if None)

        Returns:
            Dictionary of computed attributions
        """
        # Clone input tensor and ensure it requires gradients
        image = image.to(device)
        image = image.clone().detach().requires_grad_(True)
        # Check if image already has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
        print(f"Input tensor shape: {image.shape}")

        # Forward pass
        with torch.no_grad():
            output = self.model(image)

        # Get predicted class if target is not specified
        if target is None:
            target = output.argmax(dim=1).item()
            print(f"Using predicted class: {target}")

        # Available attribution methods to try
        attribution_methods = []

        # Try Saliency
        print("Calculating Saliency...")
        try:
            saliency = Saliency(self.model)
            attribution_methods.append(("Saliency", saliency))
        except Exception as e:
            print(f"Saliency not available: {e}")

        # Try Occlusion
        print("Calculating Occlusion...")
        try:
            occlusion = Occlusion(self.model)
            attribution_methods.append(("Occlusion", occlusion))
        except Exception as e:
            print(f"Occlusion not available: {e}")

        # Try Integrated Gradients
        print("Calculating Integrated Gradients...")
        try:
            integrated_grads = IntegratedGradients(self.model)
            attribution_methods.append(("Integrated Gradients", integrated_grads))
        except Exception as e:
            print(f"IntegratedGradients not available: {e}")

        # Try DeepLift
        print("Calculating DeepLift...")
        try:
            deeplift = DeepLift(self.model)
            attribution_methods.append(("DeepLift", deeplift))
        except Exception as e:
            print(f"DeepLift not available: {e}")

        # Try Guided Backpropagation
        print("Calculating Guided Backpropagation...")
        try:
            guided_backprop = GuidedBackprop(self.model)
            attribution_methods.append(("Guided Backpropagation", guided_backprop))
        except Exception as e:
            print(f"Guided Backpropagation not available: {e}")

        # Try Noise Tunnel
        print("Calculating Noise Tunnel...")
        try:
            noise_tunnel = NoiseTunnel(integrated_grads)
            attribution_methods.append(("Noise Tunnel", noise_tunnel))
        except Exception as e:
            print(f"Noise Tunnel not available: {e}")

        # If no methods are available, return
        if not attribution_methods:
            print("No attribution methods available for this model")
            return {}

        # Calculate number of plots needed
        n_methods = len(attribution_methods)

        # Plot results
        fig, axes = plt.subplots(1, 1 + n_methods, figsize=(5 * (1 + n_methods), 5))

        # Original image
        input_image = image.detach().cpu().numpy().squeeze(0)
        input_image = self.normalize_image_for_display(input_image)
        if input_image.shape[0] == 1:  # Grayscale image
            axes[0].imshow(input_image.squeeze(0), cmap="gray")
        else:  # RGB image
            axes[0].imshow(input_image.transpose(1, 2, 0))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Compute and plot attributions for each method
        attributions = {}
        for i, (method_name, method) in enumerate(attribution_methods):
            try:
                # Special handling for Occlusion which has different parameters
                if method_name == "Occlusion":
                    window_shape = (1, 16, 16) if image.shape[-1] >= 64 else (1, 3, 3)
                    attr = method.attribute(
                        image, target=target, sliding_window_shapes=window_shape
                    )
                else:
                    attr = method.attribute(image, target=target)

                # Store the attribution
                attributions[method_name] = attr

                # Process attribution for display
                attr_cpu = attr.detach().cpu().numpy().squeeze(0)
                attr_display = (
                    np.abs(attr_cpu) if method_name != "DeepLift" else attr_cpu
                )
                attr_display = self.normalize_image_for_display(attr_display)

                # Display the attribution
                if attr_display.shape[0] == 1:  # Grayscale attribution
                    axes[i + 1].imshow(attr_display.squeeze(0), cmap="hot")
                else:  # RGB attribution
                    attr_mean = np.mean(attr_display, axis=0)  # Average across channels
                    axes[i + 1].imshow(attr_mean, cmap="hot")

                axes[i + 1].set_title(method_name)
                axes[i + 1].axis("off")

            except Exception as e:
                print(f"Error calculating {method_name}: {e}")
                axes[i + 1].text(
                    0.5,
                    0.5,
                    f"Error with {method_name}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[i + 1].transAxes,
                )
                axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

        return attributions
