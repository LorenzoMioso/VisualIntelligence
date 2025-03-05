import torch
import torch.nn as nn

ADENOCARCINOMA = "adenocarcinoma"
BENIGN = "benign"
SQUAMOUS_CELL_CARCINOMA = "squamous_cell_carcinoma"
TARGET_IMAGE_SIZE = 768  # The code is written to work with 768x768 images
class_0 = BENIGN
class_1 = ADENOCARCINOMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KFOLDS = 10

# model hyperparameters
NUM_EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 0.001

DATASET_ZIP_PATH = "dataset.zip"
DATASET_PATH = "dataset"
DATASET_RESIZED_PATH = "dataset_resized"
# Output files
FILE_PATH_VAL_SPLIT = "outputs/val_split.csv"
FILE_PATH_TRAIN_SPLIT = "outputs/train_split.csv"
FILE_PATH_FOLD_STATS = "outputs/norm_stats.json"

FOLD_MODEL_RESULTS_PATH = "outputs/results_df_"
MODEL_CHECKPOINT_PATH = "outputs/checkpoint_"
MODEL_RESULT_METRICS_PATH = "outputs/metrics_"
