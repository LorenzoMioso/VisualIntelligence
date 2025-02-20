import torch
import torch.nn as nn

basedir = ""
datasetdir = "../"

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
