import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import (
    MODEL_CONFIG,
    PATH_CONFIG,
    device,
)


class TrainingMetrics:
    def __init__(self):
        with open(PATH_CONFIG.fold_stats_path, "r") as f:
            self.fold_stats = json.load(f)
        self.train_splits = pd.read_csv(PATH_CONFIG.train_split_path)
        self.val_splits = pd.read_csv(PATH_CONFIG.val_split_path)

    def show_training_results(self, results_from_csv):
        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(
            results_from_csv["epochs"],
            results_from_csv["train_loss"],
            label="train_loss",
        )
        plt.plot(
            results_from_csv["epochs"], results_from_csv["val_loss"], label="val_loss"
        )
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
            results_from_csv["epochs"],
            results_from_csv["val_acc"],
            label="val_accuracy",
        )
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

    def _f1_score(self, y_true, y_pred):
        tp = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        return f1

    def _accuracy_score(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    def compute_fold_metrics(self, model, dataloader):
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

        accuracy = self._accuracy_score(y_true, y_pred)
        f1 = self._f1_score(y_true, y_pred)
        return accuracy, f1

    def compute_metrics_all_folds(self, model_loader, dataloader_factory, model_class):
        accuracies = []
        f1_scores = []

        for i in range(10):
            model = model_loader.load_checkpoint(
                f"{PATH_CONFIG.model_checkpoint_path}{i}_{model_class.__name__}.pth"
            )
            train_idx = self.train_splits[f"train_{i}"].values
            val_idx = self.val_splits[f"val_{i}"].values
            mean = self.fold_stats[str(i)]["mean"]
            std = self.fold_stats[str(i)]["std"]

            _, val_loader = dataloader_factory.create_dataloaders(
                train_idx, val_idx, mean, std
            )

            accuracy, f1 = self.compute_fold_metrics(model, val_loader)
            accuracies.append(accuracy)
            f1_scores.append(f1)

            print(f"Fold {i} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        mean_accuracy = np.mean(accuracies)
        mean_f1_score = np.mean(f1_scores)
        print(f"\nMean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean F1 Score: {mean_f1_score:.4f}")

        result = {
            "accuracies": accuracies,
            "f1_scores": f1_scores,
            "mean_accuracy": mean_accuracy,
            "mean_f1_score": mean_f1_score,
        }

        # save to file
        with open(
            f"{PATH_CONFIG.model_result_metrics_path}{model_class.__name__}.json", "w"
        ) as f:
            json.dump(result, f)

        return result
