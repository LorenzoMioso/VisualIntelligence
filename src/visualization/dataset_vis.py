import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageFile
from torch import Tensor
from torchvision import transforms
from tqdm.auto import tqdm

from src.config import DATASET_PATH, DATASET_RESIZED_PATH, TARGET_IMAGE_SIZE
from src.dataset import DatasetCreator


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

    def visualize_class_colors(self, benign_color, cancer_color):
        """
        Visualizza i colori medi delle classi
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.imshow([[benign_color / 255]])
        ax1.set_title("Benign Average Color")
        ax1.axis("off")

        ax2.imshow([[cancer_color / 255]])
        ax2.set_title("Cancer Average Color")
        ax2.axis("off")

        plt.show()

    def get_random_image(
        self, idx=None, dataset_path=DATASET_PATH, tensor=False
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

    def inspect_image_randomly(self):
        image, label, idx = self.get_random_image(dataset_path=DATASET_PATH)
        plt.imshow(image)
        plt.title(self.df.iloc[idx]["class"])
        plt.axis("off")
        plt.show()

    def find_unique_shapes(self):
        # Finding all the unique shapes of the images inside the dataset
        shapes = []
        for idx in tqdm(range(len(self.df))):
            image = Image.open(
                f"{DATASET_PATH}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            )
            shapes.append(image.size)

        shapes = np.array(shapes)
        unique_shapes = np.unique(shapes, axis=0)
        print(f"Unique shapes: {unique_shapes}")  # 768x768

    def show_augmented_image(self):
        fold_stats = DatasetCreator().compute_fold_standardization_params()
        # read as gray scale
        image, label, idx = self.get_random_image(dataset_path=DATASET_RESIZED_PATH)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title(self.df.iloc[idx]["class"])
        ax[0].axis("off")

        # Display the image after applying the transforms
        transform = transforms.Compose(
            [
                # also add brightness change
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=fold_stats["0"]["mean"], std=fold_stats["0"]["std"]
                ),
            ]
        )

        transformed_image = transform(image)
        transformed_image = transformed_image.permute(1, 2, 0)

        ax[1].imshow(transformed_image)
        ax[1].set_title("Transformed image")
        ax[1].axis("off")
        plt.show()


if __name__ == "__main__":
    pass
