import random

import pandas as pd
from torchvision import transforms

from src.config import (
    DATASET_RESIZED_PATH,
    FOLD_MODEL_RESULTS_PATH,
    MODEL_CHECKPOINT_PATH,
    TARGET_IMAGE_SIZE,
    device,
)
from src.dataset import DataloaderFactory, DatasetCreator
from src.models.cnn import CNNImageClassifier
from src.models.scatnet import ScatNetImageClassifier
from src.models.utils import ModelAnalyzer
from src.training.metrics import TrainingMetrics
from src.training.training import CrossValidationTrainer
from src.visualization.dataset_vis import DatasetVisualizer
from src.visualization.xai import XAI
from src.visualization.xai_captum import XAI_CAPTUM


def main(model_class):
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
    mean = stats["0"]["mean"]  # type: ignore
    std = stats["0"]["std"]  # type: ignore

    train_loader, val_loader = DataloaderFactory(df).create_dataloaders(
        train_idx, val_idx, mean, std, batch_size=64, image_size=TARGET_IMAGE_SIZE
    )

    model = model_class()
    # model = ScatNetImageClassifier()
    model = model.to(device)
    model_analyzer = ModelAnalyzer(model, train_loader, val_loader)
    model_analyzer.inspect_model_architecture()
    # model_analyzer.profile_model()

    # train model
    cv_trainer = CrossValidationTrainer(model)
    # cv_trainer.train_fold(0, df, train_splits, val_splits, stats)
    cv_trainer.train_all_folds(df)

    results_df_name = f"{FOLD_MODEL_RESULTS_PATH}{0}_{model.__class__.__name__}.csv"

    results_from_csv = pd.read_csv(results_df_name)

    tm = TrainingMetrics()
    # tm.show_training_results(results_from_csv)

    res = tm.compute_metrics_all_folds(
        model_analyzer, DataloaderFactory(df), model.__class__
    )
    print(res)

    # test model
    # xai
    pass


def compute_results(model_class):
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    model = ModelAnalyzer(None, None).load_checkpoint(
        f"{MODEL_CHECKPOINT_PATH}1_{model_class.__name__}.pth"
    )
    # Assicurarsi che il modello sia sulla GPU
    model = model.to(device)
    train_splits, val_splits = dataset.create_splits()
    stats = dataset.get_standardization_params_from_file()
    if not stats:
        dataset.compute_fold_standardization_params()
    stats = dataset.get_standardization_params_from_file()
    train_idx = train_splits["train_0"].values
    val_idx = val_splits["val_0"].values
    mean = stats["0"]["mean"]  # type: ignore
    std = stats["0"]["std"]  # type: ignore

    train_loader, val_loader = DataloaderFactory(df).create_dataloaders(
        train_idx, val_idx, mean, std, batch_size=32, image_size=TARGET_IMAGE_SIZE
    )
    model_analyzer = ModelAnalyzer(model, train_loader, val_loader)

    tm = TrainingMetrics()
    res = tm.compute_metrics_all_folds(
        model_analyzer, DataloaderFactory(df), model.__class__
    )
    print(res)


def test_xai(model_class):
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    model = ModelAnalyzer(None, None).load_checkpoint(
        f"{MODEL_CHECKPOINT_PATH}1_{model_class.__name__}.pth"
    )
    # Assicurarsi che il modello sia sulla GPU
    model = model.to(device)

    xai = XAI(model)
    # xai.show_conv_filters()
    # xai.show_conv_activations(val_loader)

    # Ottieni l'immagine
    image, label, idx = DatasetVisualizer(df).get_random_image(
        tensor=True, dataset_path=DATASET_RESIZED_PATH, idx=4981
    )
    # apply transformations
    stats = dataset.get_standardization_params_from_file()
    mean = stats["0"]["mean"]  # type: ignore
    std = stats["0"]["std"]  # type: ignore
    val_transform = transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    image = val_transform(image)

    xai.backpropagation(image)
    xai.guided_backpropagation(image)


def test_model_attributions(model_class):
    """Test le diverse funzioni di visualizzazione delle attributions del modello"""

    # Carica il modello
    # (usa lo stesso codice che già usi per caricare il modello)
    df = DatasetCreator().init(TARGET_IMAGE_SIZE)
    print("df", df.head())
    model = ModelAnalyzer(None, None).load_checkpoint(
        f"{MODEL_CHECKPOINT_PATH}1_{model_class.__name__}.pth"
    )

    # Carica un'immagine di test

    image, label, idx = DatasetVisualizer(df).get_random_image(
        tensor=True, dataset_path=DATASET_RESIZED_PATH, idx=0
    )
    print("image shape dopo device", image.shape)  # type: ignore
    print("image label", label)

    # Visualizza le attributions utilizzando la funzione robusta
    attributions = XAI_CAPTUM(model).visualize_model_attributions(image)


if __name__ == "__main__":

    model_class = CNNImageClassifier
    # model_class = ScatNetImageClassifier
    # main(model_class)
    # compute_results(model_class)
    test_xai(model_class)
    # test_model_attributions(model_class)
