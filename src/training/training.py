import json
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import MODEL_CONFIG, PATH_CONFIG, device
from src.dataset import DataManager
from src.models.utils import ModelAnalyzer


class Trainer:
    """Base trainer for model training and evaluation"""

    def __init__(self, model: nn.Module, loss_fn: Optional[nn.Module] = None):
        """
        Initialize the trainer

        Args:
            model: PyTorch model to train
            loss_fn: Loss function to use
        """
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss().to(device)
        self.device = device
        self.model_analyzer = ModelAnalyzer(model)

    def train_step(
        self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int
    ) -> Tuple[float, float]:
        """
        Execute one training epoch

        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer to use
            epoch: Current epoch number

        Returns:
            Tuple of (average training loss, average training accuracy)
        """
        self.model.train()
        train_loss, train_acc = 0, 0

        for _, (img, label) in enumerate(dataloader):
            # Learning rate warmup
            warmup_percent = min(epoch / 3, 1.0)
            current_lr = 5e-4 * warmup_percent
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Move data to device
            X = img.to(self.device)
            y = label.to(self.device)

            # Forward pass
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Return average loss and accuracy
        return train_loss / len(dataloader), train_acc / len(dataloader)

    def val_step(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation data

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Tuple of (average validation loss, average validation accuracy)
        """
        self.model.eval()
        val_loss, val_acc = 0, 0

        with torch.inference_mode():
            for _, (img, label) in enumerate(dataloader):
                # Move data to device
                X = img.to(self.device)
                y = label.to(self.device)

                # Forward pass
                val_pred_logits = self.model(X)
                loss = self.loss_fn(val_pred_logits, y)
                val_loss += loss.item()

                # Calculate accuracy
                val_pred_labels = val_pred_logits.argmax(dim=1)
                val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

        # Return average loss and accuracy
        return val_loss / len(dataloader), val_acc / len(dataloader)

    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Create a learning rate scheduler

        Args:
            optimizer: Optimizer to schedule

        Returns:
            Learning rate scheduler
        """
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = MODEL_CONFIG.num_epochs,
        split: int = 0,
        patience: int = MODEL_CONFIG.patience,
        min_delta: float = MODEL_CONFIG.min_delta,
    ) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            optimizer: Optimizer to use
            epochs: Number of epochs to train for
            split: Fold number for cross-validation
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in validation loss to be considered an improvement

        Returns:
            Dictionary containing training metrics for all epochs
        """
        # Initialize tracking variables
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_without_improvement = 0

        # Setup scheduler
        scheduler = self.create_scheduler(optimizer)

        # Training loop
        for epoch in tqdm(range(epochs)):
            # Training and validation steps
            train_loss, train_acc = self.train_step(train_dataloader, optimizer, epoch)
            val_loss, val_acc = self.val_step(val_dataloader)

            scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_best_model(
                    optimizer, split, epoch, {"val_acc": val_acc, "val_loss": val_loss}
                )

            # Logging
            self._log_progress(epoch, train_loss, train_acc, val_loss, val_acc)
            self._store_results(results, train_loss, train_acc, val_loss, val_acc)

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return results

    def _log_progress(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Log training progress to console"""
        tqdm.write(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

    def _store_results(
        self,
        results: Dict[str, List[float]],
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Store training results in dictionary"""
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    def _save_best_model(
        self,
        optimizer: torch.optim.Optimizer,
        split: int,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save the best model checkpoint"""
        checkpoint_name = f"{PATH_CONFIG.model_checkpoint_path}{split}_{self.model.__class__.__name__}.pth"
        self.model_analyzer.save_checkpoint(
            model=self.model,
            filepath=checkpoint_name,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )


class CrossValidationTrainer:
    """Handle k-fold cross-validation training"""

    def __init__(self, model: nn.Module):
        """
        Initialize the cross-validation trainer

        Args:
            model: PyTorch model to train
        """
        self.model = model
        self.trainer = Trainer(model)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create an optimizer for the model

        Returns:
            Optimizer instance
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=MODEL_CONFIG.learning_rate,
            weight_decay=MODEL_CONFIG.weight_decay,
        )

    def train_all_folds(
        self, df: pd.DataFrame, num_folds: int = MODEL_CONFIG.k_folds
    ) -> None:
        """
        Train the model on all folds

        Args:
            df: DataFrame containing dataset
            num_folds: Number of folds to train on
        """
        # Load stats for standardization
        with open(PATH_CONFIG.fold_stats_path, "r") as f:
            fold_stats = json.load(f)

        # Load train/val splits
        train_splits = pd.read_csv(PATH_CONFIG.train_split_path)
        val_splits = pd.read_csv(PATH_CONFIG.val_split_path)

        # Train on each fold
        for i in range(num_folds):
            print(f"\n{'='*50}")
            print(f"Training fold {i}/{num_folds-1}")
            print(f"{'='*50}")
            self.train_fold(i, df, train_splits, val_splits, fold_stats)

    def train_fold(
        self,
        fold_num: int,
        df: pd.DataFrame,
        train_splits: pd.DataFrame,
        val_splits: pd.DataFrame,
        fold_stats: Dict[str, Dict[str, List[float]]],
    ) -> None:
        """
        Train the model on a specific fold

        Args:
            fold_num: Fold number to train on
            df: DataFrame containing dataset
            train_splits: DataFrame containing training splits
            val_splits: DataFrame containing validation splits
            fold_stats: Dictionary containing statistics for each fold
        """
        # Get indices for this fold
        train_str = f"train_{fold_num}"
        val_str = f"val_{fold_num}"
        train_idx = train_splits[train_str].values
        val_idx = val_splits[val_str].values

        # Get normalization parameters
        mean = fold_stats[str(fold_num)]["mean"]
        std = fold_stats[str(fold_num)]["std"]

        # Create data loaders
        data_manager = DataManager()
        data_manager.df = df
        train_loader, val_loader = data_manager.create_dataloaders(
            train_idx, val_idx, mean, std, batch_size=MODEL_CONFIG.batch_size
        )

        # Create fresh optimizer for each fold
        optimizer = self.create_optimizer()

        # Train the model
        start_time = timer()
        model_results = self.trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            split=fold_num,
        )
        end_time = timer()
        training_time = end_time - start_time
        print(f"Total training time for fold {fold_num}: {training_time:.3f} seconds")

        # Save results
        self._save_fold_results(fold_num, model_results)

    def _save_fold_results(
        self, fold_num: int, results: Dict[str, List[float]]
    ) -> None:
        """
        Save training results to CSV

        Args:
            fold_num: Fold number
            results: Dictionary containing training metrics
        """
        results_df = pd.DataFrame(
            {
                "train_loss": results["train_loss"],
                "val_loss": results["val_loss"],
                "train_acc": results["train_acc"],
                "val_acc": results["val_acc"],
                "epochs": range(len(results["train_loss"])),
            }
        )

        results_df_name = f"{PATH_CONFIG.fold_model_results_path}{fold_num}_{self.model.__class__.__name__}.csv"
        results_df.to_csv(results_df_name)
        print(f"Results saved to {results_df_name}")
