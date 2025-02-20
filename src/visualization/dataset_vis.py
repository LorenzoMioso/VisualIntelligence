import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

from src.config import ADENOCARCINOMA, datasetdir
from src.data.dataset import class_0, class_1, create_dataset_df


class DatasetVisualizer:
    def __init__(self):
        self.df = create_dataset_df()

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

        for file in tqdm(files, desc=f"Processing {os.path.basename(class_path)}"):
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

        # image inspection

    def inspect_image_randomly(self, dataset_path=datasetdir + "dataset"):
        idx = random.choice(range(len(self.df)))
        image = Image.open(
            f"{dataset_path}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
        )
        plt.imshow(image)
        plt.title(self.df.iloc[idx]["class"])
        plt.axis("off")
        plt.show()
        # print metadata
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

    def find_unique_shapes(self):
        # Finding all the unique shapes of the images inside the dataset
        shapes = []
        for idx in tqdm(range(len(self.df))):
            image = Image.open(
                f"dataset/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            )
            shapes.append(image.size)

        shapes = np.array(shapes)
        unique_shapes = np.unique(shapes, axis=0)
        print(f"Unique shapes: {unique_shapes}")  # 768x768


if __name__ == "__main__":
    dv = DatasetVisualizer()
    dv.plot_class_distribution()
    plt.show()
    # Calcolo colori medi
    benign_path = "dataset/benign"
    cancer_path = f"dataset/{ADENOCARCINOMA}"
    benign_color, _ = dv.compute_class_color(f"{datasetdir}dataset/{class_0}")
    cancer_color, _ = dv.compute_class_color(f"{datasetdir}dataset/{class_1}")
    dv.visualize_class_colors(benign_color, cancer_color)
