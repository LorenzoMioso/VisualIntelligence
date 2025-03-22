import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, Rectangle


def create_cnn_basic_block_diagram():
    # Create figure and axis with vertical orientation
    fig, ax = plt.subplots(figsize=(6, 10))

    # Remove axes
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Add a title
    plt.suptitle("CNN Basic Building Blocks", fontsize=16, fontweight="bold", y=0.98)

    # Define block positions and sizes
    block_width = 2.5
    block_height = 1.5
    arrow_length = 0.7

    # Use a nicer color palette - a sequential colormap
    colors = cm.Blues(np.linspace(0.5, 0.8, 4))

    # Block positions (vertical layout - from top to bottom)
    # Starting position is higher to allow for title
    y_start = 10
    conv_pos = (1.75, y_start)
    bn_pos = (1.75, conv_pos[1] - block_height - arrow_length)
    relu_pos = (1.75, bn_pos[1] - block_height - arrow_length)
    pool_pos = (1.75, relu_pos[1] - block_height - arrow_length)

    # Draw blocks
    blocks = [
        (conv_pos, "Convolution\n2D", colors[0]),
        (bn_pos, "Batch\nNormalization", colors[1]),
        (relu_pos, "ReLU\nActivation", colors[2]),
        (pool_pos, "Max Pooling", colors[3]),
    ]

    # Draw blocks with rounded corners and better styling
    for (x, y), label, color in blocks:
        rect = Rectangle(
            (x, y),
            block_width,
            block_height,
            facecolor=color,
            edgecolor="black",
            alpha=0.9,
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + block_width / 2,
            y + block_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=3,
        )

    # Draw arrows
    arrows = [
        (
            conv_pos[0] + block_width / 2,
            conv_pos[1],
            conv_pos[0] + block_width / 2,
            bn_pos[1] + block_height,
        ),
        (
            bn_pos[0] + block_width / 2,
            bn_pos[1],
            bn_pos[0] + block_width / 2,
            relu_pos[1] + block_height,
        ),
        (
            relu_pos[0] + block_width / 2,
            relu_pos[1],
            relu_pos[0] + block_width / 2,
            pool_pos[1] + block_height,
        ),
    ]

    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            color="black",
            linewidth=2,
            mutation_scale=20,
            zorder=1,
        )
        ax.add_patch(arrow)

    # Add a subtle background to make diagram stand out
    ax.set_facecolor("#f9f9f9")

    # Add a border around the entire diagram
    plt.tight_layout(rect=[0.05, 0.02, 0.95, 0.95])

    # Save figure with higher resolution
    plt.savefig("doc/imgs/cnn_basic_block.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    create_cnn_basic_block_diagram()
