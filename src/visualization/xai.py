import random

import matplotlib.pyplot as plt
import torch
from captum.attr import Occlusion

from src.config import device


class XAI:
    def __init__(self, model):
        self.model = model

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

    def show_conv_filters(self):
        # inspect filters
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)

        # Plot weights for each conv layer
        for idx, conv_layer in enumerate(conv_layers):
            # Get weights and convert to numpy
            weights = conv_layer.weight.data.cpu().numpy()

            # Calculate grid size
            n_filters = weights.shape[0]
            n_channels = weights.shape[1]

            # Create figure
            fig, axes = plt.subplots(
                n_channels, n_filters, figsize=(n_filters * 2, n_channels * 2)
            )

            # Plot each filter's channels
            for i in range(n_filters):
                for j in range(n_channels):
                    if n_channels == 1:
                        ax = axes[i] if n_filters > 1 else axes  # type: ignore
                    else:
                        ax = axes[j, i] if n_filters > 1 else axes[j]  # type: ignore

                    img = weights[i, j]
                    ax.imshow(img, cmap="viridis")
                    ax.axis("off")

            plt.suptitle(f"Conv Layer {idx+1} Weights")
            plt.tight_layout()
            plt.show()

    def show_conv_activations(self, val_loader):

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

        # Show the image
        image = image.squeeze(0).cpu().numpy()
        image = image - image.min()
        image /= image.max()
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def backpropagation(self, image):
        # Set model to training mode to enable gradient computation
        self.model.eval()

        # Ensure model is on the correct device
        self.model = self.model.to(device)

        # Make sure image is on the same device as the model and properly formatted
        image = image.to(device)

        # Check if image already has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed

        # Create a gradient object
        image = image.clone().detach().requires_grad_(True)
        print("Image shape:", image.shape)
        print("Image device:", image.device)
        print("Model device:", next(self.model.parameters()).device)

        # Forward pass
        output = self.model(image)
        print(f"Output : {output}")

        # Get the predicted class
        predicted_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        print(f"Predicted class: {predicted_class.item()}")

        # Use the predicted class directly
        print("Output for predicted class:", output[0, predicted_class])
        output_for_class = output[0, predicted_class]
        print(f"Output for predicted class _ {output_for_class}")

        # Backward pass (we want to compute gradient with respect to input)
        output_for_class.backward()

        # show gradients
        if image.grad is None:
            print(
                "No gradients were computed. The model might not support backpropagation."
            )
            return

        gradient = image.grad.detach().cpu().numpy()
        gradient = self.normalize_image_for_display(gradient)

        # Plot the input image and gradients in a single chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the input image
        input_image = image.detach().cpu().numpy().squeeze(0)
        input_image = self.normalize_image_for_display(input_image)
        input_image = input_image.squeeze(0)

        print(f"Input image shape for plotting: {input_image.shape}")
        axes[0].imshow(input_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Input Image")
        print(f"Gradient shape: {gradient.shape}")

        # Plot each gradient channel
        grad_vis = gradient.squeeze(0)
        grad_vis = grad_vis.squeeze(0)

        print(f"Gradient shape for plotting: {grad_vis.shape}")
        axes[1].imshow(grad_vis, cmap="hot")
        axes[1].axis("off")
        axes[1].set_title("Gradient")

        plt.tight_layout()
        plt.show()

    def guided_backpropagation(self, image):
        # Set the model to evaluation mode
        self.model.eval()

        # Replace ReLU with GuidedBackpropReLU
        replace_relu_with_guided(self.model)

        # Ensure model is on correct device
        self.model = self.model.to(device)

        # Make sure image is on the same device as the model and properly formatted
        image = image.to(device)

        # Check if image already has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed

        print("Image shape for guided backprop:", image.shape)
        print("Image device:", image.device)
        print("Model device:", next(self.model.parameters()).device)

        # Clone and ensure input requires gradients
        image = image.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(image)

        # Check if gradients are available
        if not hasattr(output, "grad_fn") or output.grad_fn is None:
            print(
                "Warning: output does not have grad_fn, cannot compute guided backpropagation"
            )
            return

        # Zero all existing gradients
        self.model.zero_grad()

        # Get the predicted class
        predicted_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        print(f"Predicted class: {predicted_class.item()}")

        # Use the predicted class directly
        output_for_class = output[0, predicted_class]

        # Backward pass
        output_for_class.backward()

        # Check if gradients were computed
        if image.grad is None:
            print("No gradients were computed. Using alternative visualization method.")
            return

        # Get the gradients
        gradient = image.grad.detach().cpu().numpy()

        # Normalize the gradients
        gradient = self.normalize_image_for_display(gradient)

        # Plot the original image and gradients
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the original image
        original_image = image.detach().cpu().numpy().squeeze(0)
        original_image = self.normalize_image_for_display(original_image)

        original_image = original_image.squeeze(0)

        print(f"Original image shape for plotting: {original_image.shape}")
        axes[0].imshow(original_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Original Image")

        # Plot the gradients
        grad_vis = gradient.squeeze(0)
        grad_vis = grad_vis.squeeze(0)

        print(f"Gradient shape for plotting: {grad_vis.shape}")
        axes[1].imshow(grad_vis, cmap="hot")
        axes[1].axis("off")
        axes[1].set_title("Guided Backpropagation")

        plt.tight_layout()
        plt.show()


class GuidedBackpropReLU(torch.autograd.Function):
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
    def forward(self, input):
        return GuidedBackpropReLU.apply(input)


def replace_relu_with_guided(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, GuidedBackpropReLUModel())
        else:
            replace_relu_with_guided(module)
