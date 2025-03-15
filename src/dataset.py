import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple, Union
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
    MODEL_CONFIG,
    PATH_CONFIG,
    ADENOCARCINOMA,
    BENIGN,
    SQUAMOUS_CELL_CARCINOMA,
    device,
    class_0,
    class_1,
)


class LungCancerDataset(Dataset):
    """Dataset class for lung cancer images with caching support"""

    def __init__(
        self,
        df: pd.DataFrame,
        folder_path: str,
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 100,
    ):
        self.df = df
        self.folder_path = folder_path
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self.cache:
            image, label = self.cache[idx]
        else:
            img_name = self.df.iloc[idx]["filename"]
            img_class = self.df.iloc[idx]["class"]
            img_path = os.path.join(self.folder_path, img_class, img_name)

            # Read the image using torchvision's high-performance I/O
            data = read_file(img_path)
            image = decode_jpeg(data, device=device, mode=ImageReadMode.GRAY) / 255

            # Convert class to tensor
            label = 1 if img_class == class_1 else 0
            label = torch.tensor(label, dtype=torch.long)

            # Add to cache if there's room
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (image, label)

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label


class DataManager:
    """Unified class for dataset management, preparation and loading"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def prepare_dataset(
        self, image_size: int = MODEL_CONFIG.target_image_size
    ) -> pd.DataFrame:
        """One-stop method to prepare the dataset for use"""
        self.df = self._init_dataset(image_size)
        return self.df

    def _init_dataset(self, size: int) -> pd.DataFrame:
        """Initialize the dataset - extract if needed and process if needed"""
        if not self._dataset_exists(PATH_CONFIG.dataset_path):
            self._extract_dataset()

        self._create_dataset_df()

        if not self._dataset_exists(PATH_CONFIG.dataset_resized_path):
            self._process_dataset(size)

        return self.df

    def _dataset_exists(self, path: str, no_files: int = 10000) -> bool:
        """Check if directory exists with sufficient files"""
        if not os.path.exists(path):
            print(f"Folder not found in {path}")
            return False

        if len(list(glob.iglob(path + "/**", recursive=True))) < no_files:
            print(f"Not enough files in {path}")
            return False

        print(f"Dataset found in {path}")
        return True

    def _extract_dataset(self) -> None:
        """Extract dataset from zip file"""
        with ZipFile(PATH_CONFIG.dataset_zip_path, "r") as zip_ref:
            zip_ref.extractall()

        # move folders to the right place
        os.makedirs(PATH_CONFIG.dataset_path, exist_ok=True)
        shutil.move(ADENOCARCINOMA, PATH_CONFIG.dataset_path)
        shutil.move(SQUAMOUS_CELL_CARCINOMA, PATH_CONFIG.dataset_path)
        shutil.move(BENIGN, PATH_CONFIG.dataset_path)

    def _process_dataset(self, size: int) -> None:
        """Process images to grayscale and resize"""
        if self.df is None:
            raise ValueError("Dataset DataFrame not initialized")

        os.makedirs(PATH_CONFIG.dataset_resized_path, exist_ok=True)

        for idx in tqdm(range(len(self.df)), desc="Processing images"):
            # Read image with OpenCV
            img_path = f"{PATH_CONFIG.dataset_path}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            image = cv2.imread(img_path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize
            result = cv2.resize(gray, (size, size))

            # Save
            dest_path = f"{PATH_CONFIG.dataset_resized_path}/{self.df.iloc[idx]['class']}/{self.df.iloc[idx]['filename']}"
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            cv2.imwrite(dest_path, result)

    def _create_dataset_df(self) -> None:
        """Create a DataFrame of all images"""
        allFilesClass0 = sorted(os.listdir(f"{PATH_CONFIG.dataset_path}/{class_0}"))
        allFilesClass1 = sorted(os.listdir(f"{PATH_CONFIG.dataset_path}/{class_1}"))

        df = pd.DataFrame(columns=["filename", "class"])
        df["filename"] = allFilesClass0 + allFilesClass1
        df["class"] = [class_0] * len(allFilesClass0) + [class_1] * len(allFilesClass1)

        self.df = df

    def create_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create K-fold splits for cross-validation"""
        if self.df is None:
            raise ValueError("Dataset not initialized. Call prepare_dataset first.")

        # Create stratified folds
        skf = StratifiedKFold(
            n_splits=MODEL_CONFIG.k_folds, shuffle=True, random_state=42
        )
        skf.get_n_splits(self.df["filename"], self.df["class"])

        # Storing indexes to reuse them later
        train_indexes = []
        val_indexes = []

        for train_index, val_index in skf.split(self.df["filename"], self.df["class"]):
            train_indexes.append(train_index)
            val_indexes.append(val_index)

        # Create dataframes to store the splits
        train_splits = pd.DataFrame(
            columns=[f"train_{i}" for i in range(MODEL_CONFIG.k_folds)]
        )
        val_splits = pd.DataFrame(
            columns=[f"val_{i}" for i in range(MODEL_CONFIG.k_folds)]
        )

        for i in range(MODEL_CONFIG.k_folds):
            train_splits[f"train_{i}"] = train_indexes[i]
            val_splits[f"val_{i}"] = val_indexes[i]

        # Saving the splits
        train_splits.to_csv(PATH_CONFIG.train_split_path, index=False)
        val_splits.to_csv(PATH_CONFIG.val_split_path, index=False)

        return train_splits, val_splits

    def compute_statistics(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Compute standardization parameters for all folds"""
        train_splits, val_splits = self.create_splits()
        fold_stats = {}

        for fold in range(MODEL_CONFIG.k_folds):
            print(f"Processing fold {fold}")
            train_idx = train_splits[f"train_{fold}"].values

            # Calculate statistics for this fold
            mean, std = self._fold_z_score(train_idx)
            fold_stats[fold] = {"mean": mean, "std": std}

            # Print fold statistics
            print(f"Mean: {mean}")
            print(f"Std: {std}\n")

        # Save fold statistics
        with open(PATH_CONFIG.fold_stats_path, "w") as f:
            json.dump(
                {
                    str(k): {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                    for k, v in fold_stats.items()
                },
                f,
            )

        return fold_stats

    def _fold_z_score(
        self, train_idx: List[int], is_gray: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate mean and std for a specific fold"""
        if self.df is None:
            raise ValueError("Dataset not initialized. Call prepare_dataset first.")

        train_df = self.df.iloc[train_idx]
        train_dataset = LungCancerDataset(train_df, PATH_CONFIG.dataset_resized_path)

        channels = 1 if is_gray else 3
        channels_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        channels_squared_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        pixel_count = 0

        for data, _ in tqdm(
            DataLoader(train_dataset, batch_size=1),
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

        return mean.float(), std.float()

    def get_stats_from_file(self) -> Optional[Dict[str, Dict[str, List[float]]]]:
        """Load precomputed standardization parameters"""
        if not os.path.exists(PATH_CONFIG.fold_stats_path):
            print(f"File not found in {PATH_CONFIG.fold_stats_path}")
            return None

        with open(PATH_CONFIG.fold_stats_path, "r") as f:
            fold_stats = json.load(f)

        return fold_stats

    def create_dataloaders(
        self,
        train_idx: List[int],
        val_idx: List[int],
        mean: Union[List[float], torch.Tensor],
        std: Union[List[float], torch.Tensor],
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create dataloaders for training and validation"""
        if self.df is None:
            raise ValueError("Dataset not initialized. Call prepare_dataset first.")

        # Define transforms
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(
                    MODEL_CONFIG.target_image_size, scale=(0.8, 1.0)
                ),
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
            train_df, PATH_CONFIG.dataset_resized_path, transform=train_transform
        )

        val_dataset = LungCancerDataset(
            val_df, PATH_CONFIG.dataset_resized_path, transform=val_transform
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
