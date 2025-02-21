import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
from tqdm.auto import tqdm

from src.config import basedir, device
from training.training import load_checkpoint


def save_results(model_results):
    ## These are the results from the previous step; you will load the results in the following

    # Extract train and validation loss and accuracy at each epoch
    results = dict(list(model_results.items()))

    # Get the loss values of the results dictionary (training and validation)
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]

    # Get the accuracy values of the results dictionary (training and validation)
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))
    print("epochs: ", epochs)

    ## Save results in a csv
    results_df = pd.DataFrame(
        columns=["train_loss", "val_loss", "train_acc", "val_acc", "epochs"]
    )
    results_df["train_loss"] = train_loss
    results_df["val_loss"] = val_loss
    results_df["train_acc"] = train_acc
    results_df["val_acc"] = val_acc
    results_df["epochs"] = epochs
    results_df_name = basedir + "results_df_" + str(0) + ".csv"
    results_df.to_csv(results_df_name)

    print("df: ", results_df)


def show_training_results(results_from_csv):
    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(
        results_from_csv["epochs"], results_from_csv["train_loss"], label="train_loss"
    )
    plt.plot(results_from_csv["epochs"], results_from_csv["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        results_from_csv["epochs"],
        results_from_csv["train_acc"],
        label="train_accuracy",
    )
    plt.plot(
        results_from_csv["epochs"], results_from_csv["val_acc"], label="val_accuracy"
    )
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def compute_metrics(model, dataloader):
    df = create_dataset_df()
    with open(basedir + "fold_stats.json", "r") as f:
        fold_stats = json.load(f)
    train_splits = pd.read_csv(basedir + "train_splits.csv")
    val_splits = pd.read_csv(basedir + "val_splits.csv")

    def f1_score(y_true, y_pred):
        tp = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # Avoid division by zero
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        return f1

    def accuracy_score(y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true)
        return accuracy

    def compute_metrics(model, dataloader):
        model.eval()
        y_true = np.array([])
        y_pred = np.array([])
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                y_true = np.concatenate((y_true, labels.cpu().numpy()))
                y_pred = np.concatenate((y_pred, y_pred_class.cpu().numpy()))

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, f1

    accuracies = []
    f1_scores = []
    # Iterate through all folds (0-9)
    for i in range(10):
        model = load_checkpoint(basedir + "checkpoint_" + str(i) + ".pth")
        train_str = "train_" + str(i)
        val_str = "val_" + str(i)
        train_idx = train_splits[train_str].values
        val_idx = val_splits[val_str].values
        mean = fold_stats[str(i)]["mean"]
        std = fold_stats[str(i)]["std"]
        _, val_loader = create_dataloaders(df, train_idx, val_idx, mean, std)
        accuracy, f1 = compute_metrics(model, val_loader)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        print(f"Accuracy for split {i}: {accuracy}")
        print(f"F1 Score for split {i}: {f1}")

    mean_accuracy = np.mean(accuracies)
    mean_f1_score = np.mean(f1_scores)
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean F1 Score: {mean_f1_score}")
