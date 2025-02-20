import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
from tqdm.auto import tqdm

from data.dataset import create_dataloaders, create_dataset_df
from models.cnn import ImageClassifier
from src.config import NUM_EPOCHS, basedir, device


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through DataLoader batches
    for _, (img, label) in enumerate(dataloader):
        # Warmup
        warmup_percent = min(epoch / 3, 1.0)
        current_lr = 5e-4 * warmup_percent
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # Send data to target device
        X = img.to(device)
        y = label.to(device)
        y_pred = model(X)
        # Calculate loss
        loss = loss_fn(y_pred, y)
        # Accumulate loss
        train_loss += loss.item()
        # Optimizer zero grad
        optimizer.zero_grad()
        # Loss backward
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Optimize step with scaler
        optimizer.step()
        # Calculate and accumulate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics for average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for _, (img, label) in enumerate(dataloader):
            X = img.to(device)
            y = label.to(device)

            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)

            val_loss += loss.item()
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 50,
    split: int = 0,
    patience: int = 5,
    min_delta: float = 0.001,
):
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Initialize early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-6
    )
    # Initialize tracking for best model
    best_val = 0

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
        )

        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn
        )

        # Update learning rate based on validation loss
        scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save best model
        if val_acc > best_val:
            best_val = val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            checkpoint_name = basedir + "checkpoint_" + str(split) + ".pth"
            torch.save(checkpoint, checkpoint_name)

        # Print training progress
        tqdm.write(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return results


def train_all_folds(model):
    ## You can find the model and result corresponding to each fold in the files inside the directory
    df = create_dataset_df()
    with open(basedir + "fold_stats.json", "r") as f:
        fold_stats = json.load(f)
    print(fold_stats)
    train_splits = pd.read_csv(basedir + "train_splits.csv")
    val_splits = pd.read_csv(basedir + "val_splits.csv")

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Continue the training and validation of the model for all the other folds
    for i in range(1, 10):
        # Keeping the split
        train_str = "train_" + str(i)
        val_str = "val_" + str(i)
        train_idx = train_splits[train_str].values
        val_idx = val_splits[val_str].values
        mean = fold_stats[str(i)]["mean"]
        std = fold_stats[str(i)]["std"]

        # Create dataloaders with fold-specific normalization
        train_loader, val_loader = create_dataloaders(df, train_idx, val_idx, mean, std)

        # Start the timer
        from timeit import default_timer as timer

        start_time = timer()
        # Train model
        model_results = train(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=NUM_EPOCHS,
            split=i,
        )
        # End the timer and print out how long it took
        end_time = timer()
        print(f"Total training time for split {i}: {end_time-start_time:.3f} seconds")
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
        # Save results in a csv
        results_df = pd.DataFrame(
            columns=["train_loss", "val_loss", "train_acc", "val_acc", "epochs"]
        )
        results_df["train_loss"] = train_loss
        results_df["val_loss"] = val_loss
        results_df["train_acc"] = train_acc
        results_df["val_acc"] = val_acc
        results_df["epochs"] = epochs
        results_df_name = basedir + "results_df_" + str(i) + ".csv"
        results_df.to_csv(results_df_name)


# Loading the checkpoint
def load_checkpoint(filepath):
    # Carichiamo i checkpoint in modalità sicura specificando weights_only=False
    checkpoint = torch.load(
        filepath, map_location=torch.device(device), weights_only=False
    )
    model = ImageClassifier().to(device)  # Creiamo prima il modello
    model.load_state_dict(checkpoint["state_dict"])  # Carichiamo solo i pesi
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
