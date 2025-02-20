import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
from tqdm.auto import tqdm

from src.config import basedir, device
from training.training import load_checkpoint


def show_conv_filters(model):
    # inspect filters
    conv_layers = []
    for module in model.modules():
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


def show_conv_activations(model, image):
    from captum.attr import Occlusion

    # Load the model
    model = load_checkpoint(basedir + "checkpoint_9.pth")

    # Get a single image from the validation set
    images, labels = next(iter(val_loader))
    random_idx = random.randint(0, len(images) - 1)
    image = images[random_idx].unsqueeze(0).to(device)
    label = labels[random_idx].unsqueeze(0).to(device)

    # Create an Occlusion object
    occlusion = Occlusion(model)

    # Compute the attribution
    attribution = occlusion.attribute(
        image, target=label, sliding_window_shapes=(1, 32, 32)
    )
