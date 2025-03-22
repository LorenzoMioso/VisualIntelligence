import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageFile
from torch import Tensor
from torchvision import transforms
from tqdm.auto import tqdm

from src.config import MODEL_CONFIG, PATH_CONFIG
from src.dataset import DataManager


class DatasetVisualizer:
    def __init__(self, dataset_df):
        self.df = dataset_df

    def plot_class_distribution(self):
        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.df, x="class")
        plt.title("Class distribution")

    def compute_class_color(self, class_path):
        """
        Calcola il colore medio per una classe di immagini
        """
        rgb_means = []
        files = os.listdir(class_path)

        for file in tqdm(files, desc=f"Processing color for {class_path}"):
            img_path = os.path.join(class_path, file)
            img = np.array(Image.open(img_path))
            rgb_mean = np.mean(img, axis=(0, 1))
            rgb_means.append(rgb_mean)

        class_mean = np.mean(rgb_means, axis=0)
        class_std = np.std(rgb_means, axis=0)
        return class_mean, class_std

    def show_class_colors(self, benign_color, cancer_color, vertical=False):
        """
        Visualizza i colori medi delle classi

        Args:
            benign_color: RGB color values for benign class
            cancer_color: RGB color values for cancer class
            vertical: If True, stacks the images vertically instead of horizontally
        """
        if vertical:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow([[benign_color / 255]])
        ax1.text(
            0.5,
            0.5,
            "Benign Average Color",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes,
            color="black",
            fontsize=12,
            fontweight="bold",
        )
        ax1.axis("off")

        ax2.imshow([[cancer_color / 255]])
        ax2.text(
            0.5,
            0.5,
            "Cancer Average Color",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            color="black",
            fontsize=12,
            fontweight="bold",
        )
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    def get_dataset_image(
        self, idx=None, dataset_path=PATH_CONFIG.dataset_orig_path, tensor=False
    ) -> tuple[Image.Image | Tensor, str, int]:
        if idx is None:
            idx = random.choice(range(len(self.df)))
        image = Image.open(
            f"{dataset_path}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
        )
        label = self.df.iloc[idx]["class"]
        if tensor:
            transform = transforms.Compose([transforms.ToTensor()])
            image = transform(image)
        return image, label, idx

    def show_dataset_image(self, idx=None):
        image, label, idx = self.get_dataset_image(
            dataset_path=PATH_CONFIG.dataset_orig_path, idx=idx
        )
        plt.imshow(image)
        plt.title(self.df.iloc[idx]["class"])
        plt.axis("off")
        plt.show()

    def find_unique_shapes(self):
        # Finding all the unique shapes of the images inside the dataset
        shapes = []
        for idx in tqdm(range(len(self.df))):
            image = Image.open(
                f"{PATH_CONFIG.dataset_orig_path}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            )
            shapes.append(image.size)

        shapes = np.array(shapes)
        unique_shapes = np.unique(shapes, axis=0)
        print(f"Unique shapes: {unique_shapes}")  # 768x768

    def show_augmented_image(self, idx=None):
        fold_stats = DataManager().get_stats_from_file()

        # Get image using provided or random index
        image, label, idx = self.get_dataset_image(
            idx=idx, dataset_path=PATH_CONFIG.dataset_path
        )

        # Display only the transformed image
        fig, ax = plt.subplots(figsize=(8, 8))

        # Apply the transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(
                    MODEL_CONFIG.target_image_size, scale=(0.8, 1.0)
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ElasticTransform(alpha=30.0),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
                transforms.Normalize(
                    mean=fold_stats["0"]["mean"], std=fold_stats["0"]["std"]  # type: ignore
                ),
            ]
        )

        transformed_image = transform(image)
        transformed_image = transformed_image.permute(1, 2, 0)  # type: ignore

        ax.imshow(transformed_image, cmap="gray")
        ax.set_title(f"Augmented {self.df.iloc[idx]['class']} image")
        ax.axis("off")
        plt.show()


if __name__ == "__main__":
    pass
