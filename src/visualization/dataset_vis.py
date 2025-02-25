import glob
import json
import os
import random
import shutil
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode
from torchvision.io.image import decode_jpeg, read_file
from tqdm.auto import tqdm

from dataset import class_0, class_1
from src.config import (
    ADENOCARCINOMA,
    BENIGN,
    DATASET_PATH,
    DATASET_RESIZED_PATH,
    KFOLDS,
    SQUAMOUS_CELL_CARCINOMA,
    TARGET_IMAGE_SIZE,
    class_0,
    class_1,
    device,
)
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

    def inspect_image_randomly(self):
        idx = random.choice(range(len(self.df)))
        image = Image.open(
            f"{DATASET_PATH}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
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
                f"{DATASET_PATH}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            )
            shapes.append(image.size)

        shapes = np.array(shapes)
        unique_shapes = np.unique(shapes, axis=0)
        print(f"Unique shapes: {unique_shapes}")  # 768x768

    def show_augmented_image(self):
        fold_stats = DatasetCreator().compute_fold_standardization_params()
        idx = random.choice(range(len(self.df)))
        # read as gray scale
        image = Image.open(
            f"{DATASET_RESIZED_PATH}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
        )
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
