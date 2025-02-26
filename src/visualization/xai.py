import random

import matplotlib.pyplot as plt
import torch
from captum.attr import Occlusion

from src.config import device


class XAI:
    def __init__(self, model):
        self.model = model

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
                        ax = axes[i] if n_filters > 1 else axes
                    else:
                        ax = axes[j, i] if n_filters > 1 else axes[j]

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
        # Create a gradient object
        image.requires_grad = True
        print("image shape: ", image.shape)
        output = self.model(image)

        # Create a tensor with the same shape as the output and set the predicted class to one
        target = torch.zeros_like(output)
        target[0, output.argmax()] = 1

        # Backward pass
        output.backward(gradient=target)

        # show gradients
        gradient = image.grad.detach().cpu().numpy()
        gradient = gradient - gradient.min()
        gradient /= gradient.max()

        # Plot the input image and gradients in a single chart
        fig, axes = plt.subplots(1, gradient.shape[1] + 1, figsize=(20, 20))

        # Plot the input image
        input_image = image.detach().cpu().numpy().squeeze(0)
        input_image = input_image - input_image.min()
        input_image /= input_image.max()
        input_image = input_image.transpose(1, 2, 0)
        axes[0].imshow(input_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Input Image")
        print(f"Gradient shape: {gradient.shape}")

        # Plot each gradient channel
        axes[1].imshow(gradient[0].squeeze(0), cmap="gray")
        axes[1].axis("off")
        axes[1].set_title(f"Gradient {1}")

        plt.tight_layout()
        plt.show()

    def guided_backpropagation(self, image):
        # Set the model to evaluation mode
        self.model.eval()

        # Replace ReLU with GuidedBackpropReLU
        replace_relu_with_guided(self.model)

        # Enable gradients for the input image
        image.requires_grad = True

        # Forward pass
        output = self.model(image)

        # Zero all existing gradients
        self.model.zero_grad()

        # Create a tensor with the same shape as the output and set the predicted class to one
        target = torch.zeros_like(output)
        target[0, output.argmax()] = 1

        # Backward pass
        output.backward(gradient=target)

        # Get the gradients
        gradient = image.grad.data.cpu().numpy()[0]

        # Normalize the gradients
        gradient = gradient - gradient.min()
        gradient /= gradient.max()

        # Plot the original image and gradients
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the original image
        original_image = image.detach().cpu().numpy().squeeze(0)
        original_image = original_image - original_image.min()
        original_image /= original_image.max()
        original_image = original_image.transpose(1, 2, 0)
        axes[0].imshow(original_image, cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Original Image")

        # Plot the gradients
        axes[1].imshow(gradient.transpose(1, 2, 0), cmap="gray")
        axes[1].axis("off")
        axes[1].set_title("Guided Backpropagation")

        plt.tight_layout()
        plt.show()


class GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(input, positive_mask)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        grad_input[positive_mask <= 0] = 0
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
