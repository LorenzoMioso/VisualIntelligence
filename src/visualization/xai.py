from colorsys import hls_to_rgb

import matplotlib.pyplot as plt
import numpy as np
import torch
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import fft2

from src.config import device


class XAI:
    """Explainable AI visualization methods for model interpretability"""

    def __init__(self, model):
        """
        Initialize the XAI module
        Args:
            model: PyTorch model to visualize
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode by default

    def normalize_image_for_display(self, image_array):
        """
        Normalize image data to [0,1] range for display
        Args:
            image_array: The image array to normalize
        Returns:
            Normalized image array
        """
        image = image_array - image_array.min()
        if image.max() > 0:
            image = image / image.max()
        return image

    def show_conv_filters(self):
        """Visualize convolutional filters from all conv layers"""
        # Find all convolutional layers
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)

        if not conv_layers:
            print("No convolutional layers found in the model.")
            return

        # Plot weights for each conv layer
        for idx, conv_layer in enumerate(conv_layers):
            # Get weights and convert to numpy
            weights = conv_layer.weight.data.cpu().numpy()

            # Calculate grid size
            n_filters, n_channels = weights.shape[0], weights.shape[1]
            total_images = n_filters * n_channels

            # Calculate the number of columns based on 4 rows
            n_cols = (
                total_images + 3
            ) // 4  # Ceiling division to ensure all images fit

            # Create figure with 4 rows
            fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 2, 8))

            # Handle case with fewer than 4 rows or very few filters
            if total_images <= 4:
                axes = axes.reshape(-1, 1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            # Plot each filter
            for i in range(n_filters):
                for j in range(n_channels):
                    # Calculate the position in the grid
                    idx_flat = i * n_channels + j
                    row_idx = idx_flat // n_cols
                    col_idx = idx_flat % n_cols

                    # Skip if we exceed 4 rows
                    if row_idx >= 4:
                        continue

                    # Handle different axes shapes
                    if total_images <= 4:
                        ax = (
                            axes[idx_flat][0]
                            if hasattr(axes[idx_flat], "__getitem__")
                            else axes[idx_flat]
                        )
                    elif n_cols == 1:
                        ax = (
                            axes[row_idx][0]
                            if hasattr(axes[row_idx], "__getitem__")
                            else axes[row_idx]
                        )
                    else:
                        ax = axes[row_idx, col_idx]

                    img = weights[i, j]
                    ax.imshow(img, cmap="viridis")
                    ax.axis("off")
                    ax.set_title(f"F{i}_C{j}", fontsize=8)

            # Hide unused subplots
            for idx_flat in range(total_images, 4 * n_cols):
                row_idx = idx_flat // n_cols
                col_idx = idx_flat % n_cols
                if row_idx < 4:  # Only consider the first 4 rows
                    if n_cols == 1:
                        ax = (
                            axes[row_idx][0]
                            if hasattr(axes[row_idx], "__getitem__")
                            else axes[row_idx]
                        )
                    else:
                        ax = axes[row_idx, col_idx]
                    ax.axis("off")
                    ax.set_visible(False)

            plt.suptitle(
                f"Conv Layer {idx+1} Filters - {n_filters} filters with {n_channels} channels each"
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            break

    def show_wavelet_filters(self):
        """
        Visualize wavelet filters used in scattering transform
        Displays filters for different scales (j) and orientations (theta)
        """
        # Parameters for filter bank
        M = 32
        J = 3
        L = 8
        filters_set = filter_bank(M, M, J, L=L)

        # Count total filters
        total_filters = J * L

        # Calculate optimal grid dimensions based on aspect ratio
        grid_width = int(np.ceil(np.sqrt(total_filters * 16 / 9)))  # Wider than tall
        grid_height = int(np.ceil(total_filters / grid_width))

        # Create figure with optimized grid
        fig = plt.figure(figsize=(12, 8))

        # Plot each filter
        i = 0
        for filter_idx, filter in enumerate(filters_set["psi"]):
            j_val = filter_idx // L  # Scale
            theta_val = filter_idx % L  # Orientation

            # Skip if we exceed total filters
            if i >= total_filters:
                break

            # Get filter and convert to visualization format
            f = filter["levels"][0]
            filter_c = fft2(f)
            filter_c = np.fft.fftshift(filter_c)

            # Create subplot in the grid
            ax = fig.add_subplot(grid_height, grid_width, i + 1)
            ax.imshow(self.colorize(filter_c))
            ax.axis("off")
            ax.set_title(f"j={j_val}, θ={theta_val}", fontsize=9)
            i += 1

        # Add overall title explaining the visualization
        plt.suptitle(
            "Wavelet Filters at Different Scales and Orientations\n"
            "Color shows complex values: hue=phase, saturation=magnitude",
            fontsize=14,
        )

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle
        plt.show()

    def show_low_pass_filter(self):
        # Low-pass filter
        M = 32
        J = 3
        L = 8
        filters_set = filter_bank(M, M, J, L=L)

        plt.figure()
        # plt.rc('text', usetex=True)
        plt.rc("font", family="serif")
        plt.axis("off")
        plt.set_cmap("gray_r")

        f = filters_set["phi"]["levels"][0]

        filter_c = fft2(f)
        filter_c = np.fft.fftshift(filter_c)
        plt.suptitle(
            ("The corresponding low-pass filter, also known as scaling " "function."),
            fontsize=13,
        )
        # filter_c = np.abs(filter_c)
        filter_c = self.colorize(filter_c)
        plt.imshow(filter_c)

        plt.tight_layout()
        plt.show()

    def colorize(self, z):
        n, m = z.shape
        c = np.zeros((n, m, 3))
        c[np.isinf(z)] = (1.0, 1.0, 1.0)
        c[np.isnan(z)] = (0.5, 0.5, 0.5)

        idx = ~(np.isinf(z) + np.isnan(z))
        A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
        A = (A + 0.5) % 1.0
        B = 1.0 / (1.0 + abs(z[idx]) ** 0.3)
        c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
        return c

    def backpropagation(self, image, show_original=True):
        """
        Compute and visualize vanilla backpropagation for model interpretation
        Args:
            image: Input image as tensor
        """
        # Prepare image
        image = self._prepare_input_image(image)

        # Forward pass
        output = self.model(image)

        # Get the predicted class
        predicted_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        print(f"Predicted class: {predicted_class.item()}")

        # Use the predicted class for backpropagation
        output_for_class = output[0, predicted_class]

        # Reset gradients
        self.model.zero_grad()

        # Backward pass
        output_for_class.backward()

        # Check for gradients
        if image.grad is None:
            print(
                "No gradients were computed. The model might not support backpropagation."
            )
            return

        # Get and normalize gradients
        gradient = image.grad.detach().cpu().numpy()
        input_image = image.detach().cpu().numpy()

        # Display the results
        self._display_attribution(
            input_image,
            gradient,
            "Input Image",
            "Gradient Attribution",
            show_original=show_original,
        )

    def guided_backpropagation(self, image, show_original=True):
        """
        Compute and visualize guided backpropagation for model interpretation
        Args:
            image: Input image as tensor
        """
        # Set model to evaluation mode
        self.model.eval()

        # Replace ReLU with GuidedBackpropReLU
        replace_relu_with_guided(self.model)

        # Prepare image
        image = self._prepare_input_image(image)

        # Forward pass
        output = self.model(image)

        # Check if gradients are available
        if not hasattr(output, "grad_fn") or output.grad_fn is None:
            print(
                "Warning: output does not have grad_fn, cannot compute guided backpropagation"
            )
            return

        # Reset gradients
        self.model.zero_grad()

        # Get the predicted class
        predicted_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        print(f"Predicted class: {predicted_class.item()}")

        # Use the predicted class for backpropagation
        output_for_class = output[0, predicted_class]

        # Backward pass
        output_for_class.backward()

        # Check for gradients
        if image.grad is None:
            print("No gradients were computed. Using alternative visualization method.")
            return

        # Get and normalize gradients
        gradient = image.grad.detach().cpu().numpy()
        input_image = image.detach().cpu().numpy()

        # Display the results
        self._display_attribution(
            input_image,
            gradient,
            "Input Image",
            "Guided Backpropagation",
            show_original=show_original,
        )

    def _prepare_input_image(self, image):
        """
        Prepare an image for gradient computation
        Args:
            image: Input image tensor
        Returns:
            Prepared image tensor with gradients enabled
        """
        # Move to device
        image = image.to(device)

        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Enable gradients
        image = image.clone().detach().requires_grad_(True)

        return image

    def _display_attribution(
        self,
        input_image,
        attribution,
        input_title="Input",
        attr_title="Attribution",
        show_original=True,
    ):
        """
        Display an input image and its attribution side by side
        Args:
            input_image: Input image array
            attribution: Attribution/gradient array
            input_title: Title for the input image
            attr_title: Title for the attribution image
            show_original: Whether to show the original image (default: True)
        """
        # Normalize images for display
        input_image = (
            self.normalize_image_for_display(input_image) if show_original else None
        )
        attribution = self.normalize_image_for_display(attribution)

        # Remove unnecessary dimensions
        if show_original and input_image is not None:
            if input_image.shape[0] == 1:
                input_image = input_image.squeeze(0)
            if len(input_image.shape) > 2 and input_image.shape[0] == 1:
                input_image = input_image.squeeze(0)

        if attribution.shape[0] == 1:
            attribution = attribution.squeeze(0)
        if len(attribution.shape) > 2 and attribution.shape[0] == 1:
            attribution = attribution.squeeze(0)

        # Create the plot - adjust based on whether original image is shown
        if show_original:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot the input image
            axes[0].imshow(input_image, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(input_title)

            # Plot the attribution
            axes[1].imshow(attribution, cmap="hot")
            axes[1].axis("off")
            axes[1].set_title(attr_title)
        else:
            # Only show attribution
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(attribution, cmap="hot")
            ax.axis("off")
            ax.set_title(attr_title)

        plt.tight_layout()
        plt.show()


class GuidedBackpropReLU(torch.autograd.Function):
    """Custom autograd function for guided backpropagation"""

    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = input.clamp(min=0)
        ctx.save_for_backward(positive_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (positive_mask,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero out gradient for negative input
        grad_input = grad_input * positive_mask
        # Zero out gradient for negative gradients
        grad_input = grad_input.clamp(min=0)
        return grad_input


class GuidedBackpropReLUModel(torch.nn.Module):
    """Wrapper module for the GuidedBackpropReLU autograd function"""

    def forward(self, input):
        return GuidedBackpropReLU.apply(input)


def replace_relu_with_guided(model):
    """
    Recursively replace all ReLU modules with GuidedBackpropReLUModel modules
    Args:
        model: PyTorch model to modify
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, GuidedBackpropReLUModel())
        else:
            replace_relu_with_guided(module)
