import json
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode
from torchvision.io.image import decode_jpeg, read_file
from tqdm.auto import tqdm

from src.config import (
    KFOLDS,
    TARGET_IMAGE_SIZE,
    basedir,
    class_0,
    class_1,
    datasetdir,
    device,
)


class LungCancerDataset(Dataset):
    def __init__(self, df, folder_path, transform=None, cache_size=100):
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


def create_dataset_df():
    allFilesClass0 = os.listdir(f"{datasetdir}dataset/{class_0}")
    allFilesClass1 = os.listdir(f"{datasetdir}dataset/{class_1}")
    df = pd.DataFrame(columns=["filename", "class"])
    df["filename"] = allFilesClass0 + allFilesClass1
    df["class"] = [class_0] * len(allFilesClass0) + [class_1] * len(allFilesClass1)
    return df


def create_splits():
    df = create_dataset_df()
    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    skf.get_n_splits(df["filename"], df["class"])

    # Storing indexes to reuse them later
    train_indexes = []
    val_indexes = []
    for train_index, val_index in skf.split(df["filename"], df["class"]):
        train_indexes.append(train_index)
        val_indexes.append(val_index)

    train_splits = pd.DataFrame(columns=[f"train_{i}" for i in range(KFOLDS)])
    val_splits = pd.DataFrame(columns=[f"val_{i}" for i in range(KFOLDS)])

    for i in range(KFOLDS):
        train_splits[f"train_{i}"] = train_indexes[i]
        val_splits[f"val_{i}"] = val_indexes[i]

    # Saving the splits
    train_splits.to_csv(basedir + "train_splits.csv", index=False)
    val_splits.to_csv(basedir + "val_splits.csv", index=False)
    return train_splits, val_splits


def calculate_fold_stats(df, train_idx, image_size=TARGET_IMAGE_SIZE, is_gray=True):
    """Calculate mean and std for a specific fold"""

    train_df = df.iloc[train_idx]
    train_dataset = LungCancerDataset(train_df, datasetdir + "dataset_resized")

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

    with open(basedir + "fold_stats.json", "w") as f:
        json.dump(fold_stats, f)

    return mean.float(), std.float()


def calculate_stats(df, image_size=TARGET_IMAGE_SIZE, is_gray=True):
    train_splits, val_splits = create_splits()
    fold_stats = {}
    for fold in range(KFOLDS):
        print(f"Processing fold {fold}")
        train_idx = train_splits[f"train_{fold}"].values
        val_idx = val_splits[f"val_{fold}"].values

        # Calculate statistics for this fold
        mean, std = calculate_fold_stats(df, train_idx)
        fold_stats[fold] = {"mean": mean, "std": std}

        # Print fold statistics
        print(f"Mean: {mean}")
        print(f"Std: {std}\n")

    ## Save fold statistics
    print("Stats: ", fold_stats)
    print("Saving fold statistics...")
    ## delete old file
    with open(basedir + "fold_stats.json", "w") as f:
        json.dump(
            {
                k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
                for k, v in fold_stats.items()
            },
            f,
        )


def create_dataloaders(
    df, train_idx, val_idx, mean, std, batch_size=32, image_size=TARGET_IMAGE_SIZE
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
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = LungCancerDataset(
        train_df, datasetdir + "dataset_resized", transform=train_transform
    )
    val_dataset = LungCancerDataset(
        val_df, datasetdir + "dataset_resized", transform=val_transform
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


def show_augmented_image():
    df = create_dataset_df()
    fold_stats = calculate_fold_stats(df, range(len(df)))
    idx = random.choice(range(len(df)))
    # read as gray scale
    image = Image.open(
        f"{datasetdir}dataset_resized/{df.iloc[idx]['class']}/{df.iloc[idx]['filename']}"
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title(df.iloc[idx]["class"])
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
