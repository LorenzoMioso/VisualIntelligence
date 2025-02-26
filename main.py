import random

import pandas as pd

from src.config import (
    FOLD_MODEL_RESULTS_PATH,
    MODEL_CHECKPOINT_PATH,
    TARGET_IMAGE_SIZE,
    device,
)
from src.dataset import DataloaderFactory, DatasetCreator
from src.models.cnn import CNNImageClassifier
from src.models.utils import ModelAnalyzer
from src.training.metrics import TrainingMetrics
from src.training.training import CrossValidationTrainer
from src.visualization.xai import XAI


def test_xai():
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    train_splits, val_splits = dataset.create_splits()
    stats = dataset.get_standardization_params_from_file()
    if not stats:
        dataset.compute_fold_standardization_params()
    stats = dataset.get_standardization_params_from_file()
    train_idx = train_splits["train_0"].values
    val_idx = val_splits["val_0"].values
    mean = stats["0"]["mean"]
    std = stats["0"]["std"]
    train_loader, val_loader = DataloaderFactory(df).create_dataloaders(
        train_idx, val_idx, mean, std, batch_size=32, image_size=TARGET_IMAGE_SIZE
    )
    model = ModelAnalyzer(None, None).load_checkpoint(f"{MODEL_CHECKPOINT_PATH}0.pth")

    xai = XAI(model)
    # xai.show_conv_filters()
    # xai.show_conv_activations(val_loader)

    images, labels = next(iter(val_loader))
    random_index = random.randint(0, len(images) - 1)
    xai.backpropagation(images[random_index].unsqueeze(0))
    xai.guided_backpropagation(images[random_index].unsqueeze(0))


def main():
    # process dataset
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    train_splits, val_splits = dataset.create_splits()
    stats = dataset.get_standardization_params_from_file()
    if not stats:
        dataset.compute_fold_standardization_params()
    stats = dataset.get_standardization_params_from_file()

    # first fold dataloader
    train_idx = train_splits["train_0"].values
    val_idx = val_splits["val_0"].values
    mean = stats["0"]["mean"]
    std = stats["0"]["std"]

    train_loader, val_loader = DataloaderFactory(df).create_dataloaders(
        train_idx, val_idx, mean, std, batch_size=32, image_size=TARGET_IMAGE_SIZE
    )

    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")

    model = CNNImageClassifier()
    model = model.to(device)
    model_analyzer = ModelAnalyzer(model, train_loader, val_loader)
    model_analyzer.inspect_model_architecture()
    model_analyzer.profile_model()

    # train model
    cv_trainer = CrossValidationTrainer(model)
    cv_trainer.train_all_folds(df)
    # cv_trainer.train_fold(0, df, train_splits, val_splits, stats)

    results_df_name = f"{FOLD_MODEL_RESULTS_PATH}{0}.csv"

    results_from_csv = pd.read_csv(results_df_name)

    tm = TrainingMetrics()
    print("showing training results")
    # tm.show_training_results(results_from_csv)

    acc, f1 = tm.compute_fold_metrics(model, val_loader)
    print(f"Accuracy: {acc}, F1: {f1}")

    # test model
    # xai
    pass


if __name__ == "__main__":
    # main()
    test_xai()
