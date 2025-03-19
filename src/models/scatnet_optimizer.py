import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.config import MODEL_CONFIG, PATH_CONFIG, device
from src.dataset import DataManager
from src.models.scatnet import ScatNetImageClassifier
from src.training.training import Trainer


class ScatNetOptimizer:
    """Class for finding optimal ScatNet hyperparameters using grid search"""

    def __init__(self, fold_id=0):
        """
        Initialize the ScatNet optimizer

        Args:
            fold_id: Which fold to use for optimization (default: 0)
        """
        self.fold_id = fold_id
        self.data_manager = DataManager()
        self.df = self.data_manager.prepare_dataset(MODEL_CONFIG.target_image_size)

        # Load train/val splits
        self.train_splits = pd.read_csv(PATH_CONFIG.train_split_path)
        self.val_splits = pd.read_csv(PATH_CONFIG.val_split_path)

        # Load normalization stats
        with open(PATH_CONFIG.fold_stats_path, "r") as f:
            self.fold_stats = json.load(f)

        # Get train and validation indices for this fold
        self.train_idx = self.train_splits[f"train_{fold_id}"].values
        self.val_idx = self.val_splits[f"val_{fold_id}"].values

        # Get normalization parameters for this fold
        self.mean = self.fold_stats[str(fold_id)]["mean"]
        self.std = self.fold_stats[str(fold_id)]["std"]

        # Create dataloaders
        self.train_loader, self.val_loader = self.data_manager.create_dataloaders(
            self.train_idx,
            self.val_idx,
            self.mean,
            self.std,
            batch_size=MODEL_CONFIG.batch_size,
        )

    def evaluate_model(
        self, J: int, L: int, M: int, epochs: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluate a ScatNet model with specific parameter values

        Args:
            J: Number of scales
            L: Number of orientations
            M: Maximum scattering order
            epochs: Number of epochs to train for

        Returns:
            Tuple of (best validation accuracy, best validation loss)
        """
        print(f"Evaluating ScatNet with J={J}, L={L}, M={M}")

        # Create model with these parameters
        model = ScatNetImageClassifier(J=J, L=L, M=M).to(device)

        # Initialize trainer
        trainer = Trainer(model)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=MODEL_CONFIG.learning_rate,
        )

        # Train for the specified number of epochs
        results = trainer.train(
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            optimizer=optimizer,
            epochs=epochs,
            patience=epochs,  # No early stopping for parameter search
        )

        # Get the best validation accuracy and corresponding loss
        best_val_acc = max(results["val_acc"])
        best_idx = results["val_acc"].index(best_val_acc)
        best_val_loss = results["val_loss"][best_idx]

        return best_val_acc, best_val_loss

    def grid_search(
        self,
        j_values: List[int] = [2, 3, 4, 6, 8],
        l_values: List[int] = [2, 4, 6, 8, 12],
        m_values: List[int] = [1, 2, 3],
        epochs: int = 1,
    ) -> Dict:
        """
        Perform grid search over J, L, and M parameters

        Args:
            j_values: List of J values to try
            l_values: List of L values to try
            m_values: List of M values to try
            epochs: Number of epochs for each evaluation

        Returns:
            Dictionary with grid search results
        """
        results = []

        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(j_values, l_values, m_values))

        # Evaluate each combination
        for J, L, M in tqdm(param_combinations, desc="Grid Search Progress"):
            # Check if the combination is valid
            if J <= 0 or L <= 0 or M <= 0:
                print(f"Skipping invalid combination: J={J}, L={L}, M={M}")
                continue

            try:
                # Evaluate the model with these parameters
                val_acc, val_loss = self.evaluate_model(J, L, M, epochs)

                # Store the results
                results.append(
                    {"J": J, "L": L, "M": M, "val_acc": val_acc, "val_loss": val_loss}
                )

                print(
                    f"J={J}, L={L}, M={M}: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
                )

            except Exception as e:
                print(f"Error evaluating parameters J={J}, L={L}, M={M}: {e}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save results to CSV
        os.makedirs(os.path.dirname(PATH_CONFIG.scatnet_params_path), exist_ok=True)
        results_df.to_csv(PATH_CONFIG.scatnet_params_path, index=False)

        # Find the best parameters
        if not results:
            return {"error": "No valid parameter combinations found"}

        best_idx = results_df["val_acc"].idxmax()
        best_params = results_df.iloc[best_idx].to_dict()

        print(
            f"\nBest parameters found: J={best_params['J']}, L={best_params['L']}, M={best_params['M']}"
        )
        print(f"Best validation accuracy: {best_params['val_acc']:.4f}")

        return best_params

    def random_search(
        self,
        num_trials: int = 10,
        j_range: Tuple[int, int] = (2, 5),
        l_range: Tuple[int, int] = (4, 16),
        m_values: List[int] = [1, 2],
        epochs: int = 1,
    ) -> Dict:
        """
        Perform random search over J, L, and M parameters

        Args:
            num_trials: Number of random parameter combinations to try
            j_range: (min, max) range for J values
            l_range: (min, max) range for L values
            m_values: List of M values to try
            epochs: Number of epochs for each evaluation

        Returns:
            Dictionary with random search results
        """
        results = []

        # Generate random parameter combinations
        for _ in tqdm(range(num_trials), desc="Random Search Progress"):
            J = np.random.randint(j_range[0], j_range[1] + 1)
            L = np.random.randint(l_range[0], l_range[1] + 1)
            M = np.random.choice(m_values)

            # Check if the combination is valid
            if J <= 0 or L <= 0 or M <= 0:
                print(f"Skipping invalid combination: J={J}, L={L}, M={M}")
                continue

            try:
                # Evaluate the model with these parameters
                val_acc, val_loss = self.evaluate_model(J, L, M, epochs)

                # Store the results
                results.append(
                    {"J": J, "L": L, "M": M, "val_acc": val_acc, "val_loss": val_loss}
                )

                print(
                    f"J={J}, L={L}, M={M}: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
                )

            except Exception as e:
                print(f"Error evaluating parameters J={J}, L={L}, M={M}: {e}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save results to CSV
        os.makedirs(os.path.dirname(PATH_CONFIG.scatnet_params_path), exist_ok=True)
        results_df.to_csv(PATH_CONFIG.scatnet_params_path, index=False)

        # Find the best parameters
        if not results:
            return {"error": "No valid parameter combinations found"}

        best_idx = results_df["val_acc"].idxmax()
        best_params = results_df.iloc[best_idx].to_dict()

        print(
            f"\nBest parameters found: J={best_params['J']}, L={best_params['L']}, M={best_params['M']}"
        )
        print(f"Best validation accuracy: {best_params['val_acc']:.4f}")

        return best_params
