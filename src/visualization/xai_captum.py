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

        # Define available attribution methods
        self.available_methods = [
            "Saliency",
            "Occlusion",
            "Integrated Gradients",
            "DeepLift",
            "Guided Backpropagation",
            "Noise Tunnel",
        ]

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

    def get_attribution_method(self, method_name):
        """
        Get an attribution method instance by name

        Args:
            method_name: Name of the attribution method

        Returns:
            Attribution method instance or None if not available
        """
        try:
            if method_name == "Saliency":
                return Saliency(self.model)
            elif method_name == "Occlusion":
                return Occlusion(self.model)
            elif method_name == "Integrated Gradients":
                return IntegratedGradients(self.model)
            elif method_name == "DeepLift":
                return DeepLift(self.model)
            elif method_name == "Guided Backpropagation":
                return GuidedBackprop(self.model)
            elif method_name == "Noise Tunnel":
                # Noise Tunnel needs Integrated Gradients as base
                integrated_grads = IntegratedGradients(self.model)
                return NoiseTunnel(integrated_grads)
            else:
                print(f"Unknown attribution method: {method_name}")
                return None
        except Exception as e:
            print(f"{method_name} attribution method not available: {e}")
            return None

    def list_available_methods(self):
        """
        Get a list of all available attribution methods

        Returns:
            List of available method names
        """
        return self.available_methods

    def visualize_model_attributions(
        self, image, methods=None, target=None, show_original=True
    ):
        """
        Visualize selected attribution methods for the model and input

        Args:
            image: Input tensor with shape [C, H, W] or [1, C, H, W]
            methods: List of attribution method names to use (if None, use all available methods)
                     Can be a string for a single method or a list of strings for multiple methods
            target: Target class (optional, will use predicted class if None)
            show_original: Whether to show the original image in the visualization (default: True)

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

        # Handle methods input: None, string, or list
        if methods is None:
            # Use all available methods if none specified
            methods = self.available_methods
        elif isinstance(methods, str):
            # Convert single method name to a list
            methods = [methods]

        print(f"Analyzing with methods: {methods}")

        # Initialize attribution methods list
        attribution_methods = []

        # Get the requested attribution methods
        for method_name in methods:
            if method_name in self.available_methods:
                print(f"Preparing {method_name}...")
                method = self.get_attribution_method(method_name)
                if method:
                    attribution_methods.append((method_name, method))
            else:
                print(
                    f"Method '{method_name}' is not supported. Available methods: {self.available_methods}"
                )

        # If no methods are available, return early
        if not attribution_methods:
            print("No valid attribution methods selected.")
            return {}

        # Calculate number of plots needed
        n_methods = len(attribution_methods)
        n_plots = n_methods + (1 if show_original else 0)

        # Plot results
        fig_width = 5 * n_plots if n_plots > 0 else 5
        fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, 5))

        # Make sure axes is always an array, even with a single plot
        if n_plots == 1:
            axes = np.array([axes])

        # Original image (if requested)
        if show_original:
            input_image = image.detach().cpu().numpy().squeeze(0)
            input_image = self.normalize_image_for_display(input_image)
            if input_image.shape[0] == 1:  # Grayscale image
                axes[0].imshow(input_image.squeeze(0), cmap="gray")
            else:  # RGB image
                axes[0].imshow(input_image.transpose(1, 2, 0))
            axes[0].set_title("Input Image")
            axes[0].axis("off")

        # Compute and plot attributions for each method
        attributions = {}
        for i, (method_name, method) in enumerate(attribution_methods):
            try:
                print(f"Calculating {method_name} attribution...")
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

                # Determine the index for plotting (offset by 1 if showing original)
                plot_idx = i + (1 if show_original else 0)

                # Display the attribution
                if attr_display.shape[0] == 1:  # Grayscale attribution
                    axes[plot_idx].imshow(attr_display.squeeze(0), cmap="hot")
                else:  # RGB attribution
                    attr_mean = np.mean(attr_display, axis=0)  # Average across channels
                    axes[plot_idx].imshow(attr_mean, cmap="hot")

                axes[plot_idx].set_title(f"Captum {method_name}")
                axes[plot_idx].axis("off")

            except Exception as e:
                plot_idx = i + (1 if show_original else 0)
                print(f"Error calculating {method_name}: {e}")
                axes[plot_idx].text(
                    0.5,
                    0.5,
                    f"Error with {method_name}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[plot_idx].transAxes,
                )
                axes[plot_idx].axis("off")

        plt.tight_layout()
        plt.show()

        return attributions
