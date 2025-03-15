import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

# Constants for class names
ADENOCARCINOMA = "adenocarcinoma"
BENIGN = "benign"
SQUAMOUS_CELL_CARCINOMA = "squamous_cell_carcinoma"

# Map class names to indices for easier reference
CLASS_MAPPING = {
    BENIGN: 0,
    ADENOCARCINOMA: 1,
    SQUAMOUS_CELL_CARCINOMA: 2,  # Keep for future multi-class support
}


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""

    num_epochs: int = 50
    patience: int = 10
    min_delta: float = 0.001
    target_image_size: int = 768  # The code is written to work with 768x768 images
    k_folds: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4


@dataclass
class PathConfig:
    """Configuration for file and directory paths"""

    dataset_zip_path: str = "dataset.zip"
    dataset_path: str = "dataset"
    dataset_resized_path: str = "dataset_resized"
    output_dir: str = "outputs"

    @property
    def val_split_path(self) -> str:
        return f"{self.output_dir}/val_split.csv"

    @property
    def train_split_path(self) -> str:
        return f"{self.output_dir}/train_split.csv"

    @property
    def fold_stats_path(self) -> str:
        return f"{self.output_dir}/norm_stats.json"

    @property
    def fold_model_results_path(self) -> str:
        return f"{self.output_dir}/results_df_"

    @property
    def model_checkpoint_path(self) -> str:
        return f"{self.output_dir}/checkpoint_"

    @property
    def model_result_metrics_path(self) -> str:
        return f"{self.output_dir}/metrics_"


# Create instances of configs for easy import
MODEL_CONFIG = ModelConfig()
PATH_CONFIG = PathConfig()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For backwards compatibility with existing code
class_0 = BENIGN
class_1 = ADENOCARCINOMA
