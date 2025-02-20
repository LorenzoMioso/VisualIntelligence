import glob
import os
import shutil
from zipfile import ZipFile

import cv2
from tqdm.auto import tqdm

from src.config import ADENOCARCINOMA, BENIGN, SQUAMOUS_CELL_CARCINOMA, datasetdir


def dataset_exists(path=datasetdir + "/dataset", no_files=10000):
    # check if dir exists and contains at least no_files files
    if not os.path.exists(path):
        print(f"Folder not found in {path}")
        return False
    if len(list(glob.iglob(path + "/**", recursive=True))) < no_files:
        print(f"Not enough files in {path}")
        return False
    print(f"Dataset found in {path}")
    return True


if not dataset_exists():
    with ZipFile(datasetdir + "dataset.zip", "r") as zip_ref:
        zip_ref.extractall()
    # move folders to the right place
    os.makedirs("dataset", exist_ok=True)
    shutil.move(ADENOCARCINOMA, "dataset")
    shutil.move(SQUAMOUS_CELL_CARCINOMA, "dataset")
    shutil.move(BENIGN, "dataset")


def resize_dataset(df, size):
    new_folder = datasetdir + "dataset_resized"
    os.makedirs(new_folder, exist_ok=True)
    for idx in tqdm(range(len(df))):
        # Read image with OpenCV
        img_path = (
            f"{datasetdir}dataset/{df.iloc[idx]['class']}/{df.iloc[idx]['filename']}"
        )
        image = cv2.imread(img_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize
        result = cv2.resize(gray, size)
        # Save
        dest_path = f"{new_folder}/{df.iloc[idx]['class']}/{df.iloc[idx]['filename']}"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        cv2.imwrite(dest_path, result)
