import json
from timeit import default_timer as timer

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
from tqdm.auto import tqdm

from src.config import (
    FILE_PATH_FOLD_STATS,
    FILE_PATH_TRAIN_SPLIT,
    FILE_PATH_VAL_SPLIT,
    FOLD_MODEL_RESULTS_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_EPOCHS,
    TARGET_IMAGE_SIZE,
    device,
)
from src.dataset import DataloaderFactory
from src.models.cnn import CNNImageClassifier
from src.models.scatnet import ScatNetImageClassifier


class Trainer:
    def __init__(self, model, loss_fn=nn.CrossEntropyLoss()):
        self.model = model
        self.loss_fn = loss_fn.to(device)
        self.device = device

    def train_step(self, dataloader, optimizer, epoch):
        self.model.train()
        train_loss, train_acc = 0, 0

        for _, (img, label) in enumerate(dataloader):
            # Warmup
            warmup_percent = min(epoch / 3, 1.0)
            current_lr = 5e-4 * warmup_percent
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            X = img.to(self.device)
            y = label.to(self.device)

            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        return train_loss / len(dataloader), train_acc / len(dataloader)

    def val_step(self, dataloader):
        self.model.eval()
        val_loss, val_acc = 0, 0

        with torch.inference_mode():
            for _, (img, label) in enumerate(dataloader):
                X = img.to(self.device)
                y = label.to(self.device)

                val_pred_logits = self.model(X)
                loss = self.loss_fn(val_pred_logits, y)
                val_loss += loss.item()

                val_pred_labels = val_pred_logits.argmax(dim=1)
                val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

        return val_loss / len(dataloader), val_acc / len(dataloader)

    def train(
        self,
        train_dataloader,
        val_dataloader,
        optimizer,
        epochs=50,
        split=0,
        patience=5,
        min_delta=0.001,
    ):
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_val = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=1e-6
        )

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(train_dataloader, optimizer, epoch)
            val_loss, val_acc = self.val_step(val_dataloader)

            scheduler.step()

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if val_acc > best_val:
                best_val = val_acc
                self.save_checkpoint(optimizer, split, self.model.__class__)

            self._log_progress(epoch, train_loss, train_acc, val_loss, val_acc)
            self._store_results(results, train_loss, train_acc, val_loss, val_acc)

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return results

    def _log_progress(self, epoch, train_loss, train_acc, val_loss, val_acc):
        tqdm.write(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

    def _store_results(self, results, train_loss, train_acc, val_loss, val_acc):
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    def save_checkpoint(
        self,
        optimizer,
        split,
        model_class,
    ):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_class": self.model.__class__.__name__,
        }
        checkpoint_name = (
            MODEL_CHECKPOINT_PATH + str(split) + "_" + model_class.__name__ + ".pth"
        )
        torch.save(checkpoint, checkpoint_name)


class CrossValidationTrainer:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=5e-4
        )
        self.model = model
        self.trainer = Trainer(model)

    def train_all_folds(self, df):
        with open(FILE_PATH_FOLD_STATS, "r") as f:
            fold_stats = json.load(f)
        print(fold_stats)

        train_splits = pd.read_csv(FILE_PATH_TRAIN_SPLIT)
        val_splits = pd.read_csv(FILE_PATH_VAL_SPLIT)

        for i in range(1, 10):
            self.train_fold(i, df, train_splits, val_splits, fold_stats)

    def train_fold(self, fold_num, df, train_splits, val_splits, fold_stats):
        train_str = f"train_{fold_num}"
        val_str = f"val_{fold_num}"
        train_idx = train_splits[train_str].values
        val_idx = val_splits[val_str].values
        mean = fold_stats[str(fold_num)]["mean"]
        std = fold_stats[str(fold_num)]["std"]

        train_loader, val_loader = DataloaderFactory(df).create_dataloaders(
            train_idx, val_idx, mean, std, batch_size=64, image_size=TARGET_IMAGE_SIZE
        )

        start_time = timer()
        model_results = self.trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=self.optimizer,
            epochs=NUM_EPOCHS,
            split=fold_num,
        )
        end_time = timer()
        print(
            f"Total training time for split {fold_num}: {end_time-start_time:.3f} seconds"
        )

        self._save_fold_results(fold_num, model_results, self.model.__class__)

    def _save_fold_results(self, fold_num, results, model_class):
        results = dict(list(results.items()))
        results_df = pd.DataFrame(
            {
                "train_loss": results["train_loss"],
                "val_loss": results["val_loss"],
                "train_acc": results["train_acc"],
                "val_acc": results["val_acc"],
                "epochs": range(len(results["train_loss"])),
            }
        )
        results_df_name = (
            f"{FOLD_MODEL_RESULTS_PATH}{fold_num}{model_class.__name__}.csv"
        )
        results_df.to_csv(results_df_name)
