import random
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import fft2


import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Occlusion

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

            # Create figure
            fig, axes = plt.subplots(
                n_channels, n_filters, figsize=(n_filters * 2, n_channels * 2)
            )

            # Plot each filter's channels
            for i in range(n_filters):
                for j in range(n_channels):
                    if n_channels == 1:
                        ax = axes[i] if n_filters > 1 else axes
                    else:
                        ax = axes[j, i] if n_filters > 1 else axes[j]
                    img = weights[i, j]
                    ax.imshow(img, cmap="viridis")
                    ax.axis("off")

            plt.suptitle(f"Conv Layer {idx+1} Filters")
            plt.tight_layout()
            plt.show()

    def show_scattering_filters(self):
        # Gray-scale filters
        M = 32
        J = 3
        L = 8
        filters_set = filter_bank(M, M, J, L=L)

        fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        # plt.rc('text', usetex=True)
        plt.rc("font", family="serif")
        i = 0
        for filter in filters_set["psi"]:
            f = filter["levels"][0]
            filter_c = fft2(f)
            filter_c = np.fft.fftshift(filter_c)
            axs[i // L, i % L].imshow(
                np.abs(filter_c), cmap="gray", vmin=0, vmax=1.5 * np.abs(filter_c).max()
            )
            axs[i // L, i % L].axis("off")
            axs[i // L, i % L].set_title("j={}\ntheta={}".format(i // L, i % L))
            i = i + 1

        fig.suptitle(
            (
                r"Wavelets for each scales j and angles theta used."
                "\nColor saturation and color hue respectively denote complex "
                "magnitude and complex phase."
            ),
            fontsize=13,
        )
        plt.tight_layout()
        plt.show()  # Changed from fig.show() to plt.show()

    def show_conv_activations(self, val_loader):
        """
        Visualize activations of convolutional layers using occlusion
        Args:
            val_loader: DataLoader with validation data
        """
        # Get a single image from the validation set
        images, labels = next(iter(val_loader))
        random_idx = random.randint(0, len(images) - 1)
        image = images[random_idx].unsqueeze(0).to(device)
        label = labels[random_idx].unsqueeze(0).to(device)

        # Create an Occlusion object
        occlusion = Occlusion(self.model)

        # Compute the attribution
        attribution = occlusion.attribute(
            image, target=label, sliding_window_shapes=(1, 32, 32)
        )

        # Display the image alongside the attribution
        self._display_attribution(
            image.detach().cpu().numpy(),
            attribution.detach().cpu().numpy(),
            "Original Image",
            "Occlusion Attribution",
        )

    def backpropagation(self, image):
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
            input_image, gradient, "Input Image", "Gradient Attribution"
        )

    def guided_backpropagation(self, image):
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
            input_image, gradient, "Original Image", "Guided Backpropagation"
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
        self, input_image, attribution, input_title="Input", attr_title="Attribution"
    ):
        """
        Display an input image and its attribution side by side
        Args:
            input_image: Input image array
            attribution: Attribution/gradient array
            input_title: Title for the input image
            attr_title: Title for the attribution image
        """
        # Normalize images for display
        input_image = self.normalize_image_for_display(input_image)
        attribution = self.normalize_image_for_display(attribution)

        # Remove unnecessary dimensions
        if input_image.shape[0] == 1:
            input_image = input_image.squeeze(0)
        if len(input_image.shape) > 2 and input_image.shape[0] == 1:
            input_image = input_image.squeeze(0)

        if attribution.shape[0] == 1:
            attribution = attribution.squeeze(0)
        if len(attribution.shape) > 2 and attribution.shape[0] == 1:
            attribution = attribution.squeeze(0)

        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the input image
        axes[0].imshow(input_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title(input_title)

        # Plot the attribution
        axes[1].imshow(attribution, cmap="hot")
        axes[1].axis("off")
        axes[1].set_title(attr_title)

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
