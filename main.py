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
from src.dataset import DataloaderFactory, DatasetCreator
from src.models.cnn import ImageClassifier, inspect_model_architecture, profile_model
from src.training.training import CrossValidationTrainer


def main():
    # process dataset
    dataset = DatasetCreator()
    df = dataset.init(TARGET_IMAGE_SIZE)
    train_splits, val_splits = dataset.create_splits()
    stats = dataset.get_standardization_params_from_file()
    if not stats:
        stats = dataset.compute_fold_standardization_params()

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

    model = ImageClassifier()
    model = model.to(device)
    # inspect_model_architecture(model, train_loader, val_loader)
    # profile_model(model, train_loader)

    # train model
    cv_trainer = CrossValidationTrainer(model)
    # cv_trainer.train_all_folds(df)
    cv_trainer.train_fold(0, df, train_splits, val_splits, stats)

    # test model
    # xai
    pass


if __name__ == "__main__":
    main()
