import glob
import json
import os
import shutil
from zipfile import ZipFile

import cv2
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode
from torchvision.io.image import decode_jpeg, read_file
from tqdm.auto import tqdm

from src.config import (
    ADENOCARCINOMA,
    BENIGN,
    KFOLDS,
    SQUAMOUS_CELL_CARCINOMA,
    TARGET_IMAGE_SIZE,
    class_0,
    class_1,
    device,
)


class DatasetCreator:
    def __init__(self):
        self.dataset_path = "./dataset"

    def _dataset_exists(self, path, no_files=10000):
        # check if dir exists and contains at least no_files files
        if not os.path.exists(self.dataset_path):
            print(f"Folder not found in {self.dataset_path}")
            return False
        if len(list(glob.iglob(self.dataset_path + "/**", recursive=True))) < no_files:
            print(f"Not enough files in {self.dataset_path}")
            return False
        print(f"Dataset found in {self.dataset_path}")
        return True

    def _extract_dataset(self):
        with ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall()
        # move folders to the right place
        os.makedirs("dataset", exist_ok=True)
        shutil.move(ADENOCARCINOMA, "dataset")
        shutil.move(SQUAMOUS_CELL_CARCINOMA, "dataset")
        shutil.move(BENIGN, "dataset")

    def resize_dataset(self, size):
        new_folder = "dataset_resized"
        os.makedirs(new_folder, exist_ok=True)
        for idx in tqdm(range(len(self.df))):
            # Read image with OpenCV
            img_path = (
                f"dataset/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            )
            image = cv2.imread(img_path)
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize
            result = cv2.resize(gray, (size, size))
            # Save
            dest_path = f"{new_folder}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            cv2.imwrite(dest_path, result)

    def _create_dataset_df(self):
        allFilesClass0 = os.listdir(f"dataset/{class_0}")
        allFilesClass1 = os.listdir(f"dataset/{class_1}")
        df = pd.DataFrame(columns=["filename", "class"])
        df["filename"] = allFilesClass0 + allFilesClass1
        df["class"] = [class_0] * len(allFilesClass0) + [class_1] * len(allFilesClass1)
        self.df = df

    def init(
        self, size, dataset_path="./dataset", dataset_resized_path="./dataset_resized"
    ):
        if not self._dataset_exists(dataset_path):
            self._extract_dataset()
        self._create_dataset_df()
        if not self._dataset_exists(dataset_resized_path):
            self.resize_dataset(size)
        return self.df

    def create_splits(self):
        skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42)
        skf.get_n_splits(self.df["filename"], self.df["class"])

        # Storing indexes to reuse them later
        train_indexes = []
        val_indexes = []
        for train_index, val_index in skf.split(self.df["filename"], self.df["class"]):
            train_indexes.append(train_index)
            val_indexes.append(val_index)

        train_splits = pd.DataFrame(columns=[f"train_{i}" for i in range(KFOLDS)])
        val_splits = pd.DataFrame(columns=[f"val_{i}" for i in range(KFOLDS)])

        for i in range(KFOLDS):
            train_splits[f"train_{i}"] = train_indexes[i]
            val_splits[f"val_{i}"] = val_indexes[i]

        # Saving the splits
        train_splits.to_csv("train_splits.csv", index=False)
        val_splits.to_csv("val_splits.csv", index=False)
        return train_splits, val_splits

    def _fold_z_score(self, train_idx, is_gray=True):
        """Calculate mean and std for a specific fold"""

        train_df = self.df.iloc[train_idx]
        train_dataset = LungCancerDataset(train_df, "dataset_resized")

        channels = 1 if is_gray else 3
        channels_sum = torch.zeros(
            channels, dtype=torch.float64, device=device
        )  # Use float64
        channels_squared_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        pixel_count = 0

        for data, _ in tqdm(
            DataLoader(
                train_dataset,
                batch_size=1,
            ),
            desc="Calculating dataset statistics",
        ):
            data = data.to(dtype=torch.float64)
            batch_pixels = data.size(0) * data.size(2) * data.size(3)
            channels_sum += torch.sum(data, dim=[0, 2, 3])
            channels_squared_sum += torch.sum(data**2, dim=[0, 2, 3])
            pixel_count += batch_pixels

        mean = channels_sum / pixel_count

        variance = (channels_squared_sum / pixel_count) - (mean**2)
        variance = torch.clamp(variance, min=1e-10)
        std = torch.sqrt(variance)

        fold_stats = {
            "0": {"mean": mean[0].item(), "std": std[0].item()},
        }

        with open("fold_stats.json", "w") as f:
            json.dump(fold_stats, f)

        return mean.float(), std.float()

    def compute_fold_standardization_params(self):
        # read
        train_splits, val_splits = self.create_splits()
        fold_stats = {}
        for fold in range(KFOLDS):
            print(f"Processing fold {fold}")
            train_idx = train_splits[f"train_{fold}"].values
            val_idx = val_splits[f"val_{fold}"].values

            # Calculate statistics for this fold
            mean, std = self._fold_z_score(train_idx)
            fold_stats[fold] = {"mean": mean, "std": std}

            # Print fold statistics
            print(f"Mean: {mean}")
            print(f"Std: {std}\n")

        ## Save fold statistics
        print("Stats: ", fold_stats)
        print("Saving fold statistics...")
        ## delete old file
        with open("fold_stats.json", "w") as f:
            json.dump(
                {
                    k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                    for k, v in fold_stats.items()
                },
                f,
            )
        return fold_stats

    def get_standardization_params_from_file(self):
        if not os.path.exists("fold_stats.json"):
            print("File fold_stats.json not found")
            return None
        with open("fold_stats.json", "r") as f:
            fold_stats = json.load(f)
        return fold_stats


class LungCancerDataset(Dataset):
    def __init__(self, df, folder_path, transform=None, cache_size=1000):
        self.df = df
        self.folder_path = folder_path
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            image, label = self.cache[idx]
        else:
            img_name = self.df.iloc[idx]["filename"]
            img_class = self.df.iloc[idx]["class"]
            img_path = os.path.join(self.folder_path, img_class, img_name)

            data = read_file(img_path)
            image = decode_jpeg(data, device=device, mode=ImageReadMode.GRAY) / 255

            label = 1 if img_class == class_1 else 0
            label = torch.tensor(label, dtype=torch.long)

            # Add to cache
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (image, label)

        if self.transform:
            image = self.transform(image)

        return image, label


class DataloaderFactory:
    def __init__(self, df):
        self.df = df

    def create_dataloaders(
        self,
        train_idx,
        val_idx,
        mean,
        std,
        batch_size=32,
        image_size=TARGET_IMAGE_SIZE,
    ):
        """Create normalized dataloaders for a specific fold"""

        # Define transforms
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(768, scale=(0.8, 1.0)),
                transforms.RandomGrayscale(p=0.1),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # Create datasets
        train_df = self.df.iloc[train_idx]
        val_df = self.df.iloc[val_idx]

        train_dataset = LungCancerDataset(
            train_df, "dataset_resized", transform=train_transform
        )
        val_dataset = LungCancerDataset(
            val_df, "dataset_resized", transform=val_transform
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_loader, val_loader
