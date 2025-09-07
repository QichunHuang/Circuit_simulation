#!/usr/bin/env python3
"""
Experiment 1: Single Dimension Perturbation Experiment

Objective: Train 5 independent networks, each using corresponding single-dimension variation dataset
- L1 variation dataset (mixed_z_dataset_0.csv) → FCNN training
- L2 variation dataset (mixed_z_dataset_1.csv) → FCNN training
- L3 variation dataset (mixed_z_dataset_2.csv) → FCNN training
- L4 variation dataset (mixed_z_dataset_3.csv) → FCNN training
- L5 variation dataset (mixed_z_dataset_4.csv) → FCNN training

Expected Results:
- Record training curves for each network
- Compare prediction difficulty across L dimensions
- Generate performance metrics summary table
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ComplexImpedanceDataset(Dataset):
    """Complex impedance dataset class for FCNN training"""

    def __init__(self, z_data, l_data, transform=None):
        """Initialize dataset

        Args:
            z_data: Complex impedance data (N_samples, 3000)
            l_data: Inductance target values (N_samples, 5)
            transform: Optional data transformation
        """
        self.z_data = torch.FloatTensor(z_data)
        self.l_data = torch.FloatTensor(l_data)
        self.transform = transform

    def __len__(self):
        return len(self.z_data)

    def __getitem__(self, idx):
        z_sample = self.z_data[idx]
        l_sample = self.l_data[idx]

        if self.transform:
            z_sample = self.transform(z_sample)

        return z_sample, l_sample


class FCNNNetwork(nn.Module):
    """FCNN network architecture"""

    def __init__(
        self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]
    ):
        """Initialize FCNN network

        Args:
            input_size: Input feature dimension
            output_size: Output dimension (number of inductance parameters)
            hidden_sizes: List of hidden layer neuron counts
        """
        super(FCNNNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.3),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


class SingleLTrainer:
    """Single dimension inductance variation experiment trainer"""

    def __init__(self, model, device="cpu", learning_rate=0.001):
        """Initialize trainer

        Args:
            model: FCNN model instance
            device: Training device
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # Training history records
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Single training epoch"""
        self.model.train()
        total_loss = 0.0

        for z_batch, l_batch in train_loader:
            z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(z_batch)
            loss = self.criterion(outputs, l_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        """Single validation epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for z_batch, l_batch in val_loader:
                z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)
                outputs = self.model(z_batch)
                loss = self.criterion(outputs, l_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=200, l_dim_name="L0"):
        """Complete training workflow

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum training epochs
            l_dim_name: Inductance dimension name (e.g., L1, L2, etc.)

        Returns:
            dict: Training history and best performance metrics
        """
        print(f"\n=== Starting training {l_dim_name} variation network ===")
        print(f"Training device: {self.device}")
        print(f"Maximum epochs: {epochs}")

        # Early stopping parameters
        patience = 20
        patience_counter = 0

        start_time = time.time()

        for epoch in tqdm(range(epochs), desc=f"Training {l_dim_name} network"):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                import copy

                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"{l_dim_name} Epoch {epoch + 1:3d}: "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}"
                )

            # Early stopping check
            if patience_counter >= patience:
                print(
                    f"{l_dim_name} early stopped at epoch {epoch + 1} (patience={patience})"
                )
                break

        training_time = time.time() - start_time

        print(f"{l_dim_name} training completed:")
        print(f"  - Best validation loss: {self.best_val_loss:.6f}")
        print(f"  - Training epochs: {len(self.train_losses)}")
        print(f"  - Training time: {training_time:.2f} seconds")

        return {
            "l_dim_name": l_dim_name,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
            "training_time": training_time,
        }

    def evaluate(self, test_loader, l_dim_name="L0"):
        """Model evaluation

        Args:
            test_loader: Test data loader
            l_dim_name: Inductance dimension name

        Returns:
            dict: Evaluation metrics
        """
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for z_batch, l_batch in test_loader:
                z_batch = z_batch.to(self.device)
                l_batch = l_batch.to(self.device)
                outputs = self.model(z_batch)

                predictions.append(outputs.cpu().numpy())
                targets.append(l_batch.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)

        print(f"\n=== {l_dim_name} Network Evaluation Results ===")
        print(f"Test MSE:  {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")
        print(f"Test R²:   {r2:.6f}")

        # Individual coil performance
        coil_names = ["L1", "L2", "L3", "L4", "L5"]
        coil_metrics = {}
        print("\n Individual coil performance:")
        for i, name in enumerate(coil_names):
            coil_mse = mean_squared_error(targets[:, i], predictions[:, i])
            coil_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            coil_r2 = r2_score(targets[:, i], predictions[:, i])
            coil_metrics[name] = {"mse": coil_mse, "mae": coil_mae, "r2": coil_r2}
            print(
                f"  {name} - MSE: {coil_mse:.6f}, MAE: {coil_mae:.6f}, R²: {coil_r2:.6f}"
            )

        return {
            "l_dim_name": l_dim_name,
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "coil_metrics": coil_metrics,
        }


def load_dataset(dataset_path, l_dim_idx):
    print(f"Loading dataset: {dataset_path}")

    # 使用Polars加载数据
    df = pl.read_csv(dataset_path)

    # 提取特征和目标
    z_columns = [
        col
        for col in df.columns
        if col.startswith("z_") and ("real" in col or "imag" in col)
    ]
    l_columns = [col for col in df.columns if col.startswith("L") and len(col) == 2]

    z_data = df.select(z_columns).to_numpy()
    l_data = df.select(l_columns).to_numpy()

    print("Dataset loaded successfully:")
    print(f"  - Complex impedance features shape: {z_data.shape}")
    print(f"  - Inductance targets shape: {l_data.shape}")
    print(f"  - Main variation dimension: L{l_dim_idx + 1}")

    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=0.2, random_state=42
    )
    z_train, z_val, l_train, l_val = train_test_split(
        z_temp, l_temp, test_size=0.25, random_state=42
    )

    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)

    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)

    print("Data splitting completed:")
    print(f"  - Training set: {len(z_train)} samples")
    print(f"  - Validation set: {len(z_val)} samples")
    print(f"  - Test set: {len(z_test)} samples")

    return (z_train, l_train), (z_val, l_val), (z_test, l_test), (z_scaler, l_scaler)


def create_data_loaders(train_data, val_data, test_data, batch_size=64):
    train_dataset = ComplexImpedanceDataset(train_data[0], train_data[1])
    val_dataset = ComplexImpedanceDataset(val_data[0], val_data[1])
    test_dataset = ComplexImpedanceDataset(test_data[0], test_data[1])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def plot_training_curves(
    all_results, save_path="results/experiment1_training_curves.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    for i, result in enumerate(all_results):
        l_name = result["l_dim_name"]
        epochs = range(1, len(result["train_losses"]) + 1)
        ax1.plot(
            epochs,
            result["train_losses"],
            label=f"{l_name} Train",
            color=colors[i],
            alpha=0.7,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Experiment 1: Training Loss Comparison across L Dimensions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    for i, result in enumerate(all_results):
        l_name = result["l_dim_name"]
        epochs = range(1, len(result["val_losses"]) + 1)
        ax2.plot(
            epochs,
            result["val_losses"],
            label=f"{l_name} Val",
            color=colors[i],
            alpha=0.7,
            linestyle="--",
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Experiment 1: Validation Loss Comparison across L Dimensions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Training curves comparison plot saved: {save_path}")


def save_experiment_results(
    all_training_results, all_eval_results, save_path="results/experiment1_results.json"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    summary = {
        "experiment_name": "experiment 1: single dimension perturbation",
        "networks_trained": len(all_training_results),
        "training_summary": [],
        "evaluation_summary": [],
    }

    for train_result, eval_result in zip(all_training_results, all_eval_results):
        train_summary = {
            "l_dim_name": train_result["l_dim_name"],
            "epochs_trained": train_result["epochs_trained"],
            "best_val_loss": train_result["best_val_loss"],
            "training_time": train_result["training_time"],
        }
        summary["training_summary"].append(train_summary)

        eval_summary = {
            "l_dim_name": eval_result["l_dim_name"],
            "test_mse": eval_result["mse"],
            "test_rmse": eval_result["rmse"],
            "test_mae": eval_result["mae"],
            "test_r2": eval_result["r2"],
            "coil_metrics": eval_result["coil_metrics"],
        }
        summary["evaluation_summary"].append(eval_summary)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Experiment results saved: {save_path}")


def main():
    """experiment 1: single dimension perturbation"""
    print("=" * 60)
    print("Experiment 1: Single Dimension Perturbation Experiment")
    print("=" * 60)
    print(
        "Objective: Train 5 independent networks, each corresponding to one L dimension variation"
    )
    print("Network architecture: FCNN (3000→1024→512→256→128→5)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    l_dim_names = ["L1", "L2", "L3", "L4", "L5"]
    dataset_files = [f"mixed_z_dataset_{i}.csv" for i in range(5)]

    all_training_results = []
    all_eval_results = []

    for i, (l_name, dataset_file) in enumerate(zip(l_dim_names, dataset_files)):
        print(f"\n{'=' * 50}")
        print(f"Starting training {l_name} variation network ({i + 1}/5)")
        print(f"{'=' * 50}")

        try:
            train_data, val_data, test_data, scalers = load_dataset(dataset_file, i)

            train_loader, val_loader, test_loader = create_data_loaders(
                train_data, val_data, test_data, batch_size=64
            )

            model = FCNNNetwork(input_size=3000, output_size=5)
            print(
                f"{l_name} network parameter count: {sum(p.numel() for p in model.parameters()):,}"
            )

            trainer = SingleLTrainer(model, device=device, learning_rate=0.001)

            training_result = trainer.train(
                train_loader, val_loader, epochs=200, l_dim_name=l_name
            )

            eval_result = trainer.evaluate(test_loader, l_dim_name=l_name)

            model_path = f"models/experiment1_{l_name.lower()}_network.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": trainer.best_model_state,
                    "model_config": {
                        "input_size": 3000,
                        "output_size": 5,
                        "hidden_sizes": [1024, 512, 256, 128],
                    },
                    "training_result": training_result,
                    "eval_result": eval_result,
                    "scalers": scalers,
                },
                model_path,
            )
            print(f"{l_name} model saved: {model_path}")

            all_training_results.append(training_result)
            all_eval_results.append(eval_result)

        except Exception as e:
            print(f"❌ {l_name} training failed: {str(e)}")
            continue

    if all_training_results:
        print(f"\n{'=' * 60}")
        print("Generating Experiment 1 results analysis")
        print(f"{'=' * 60}")

        plot_training_curves(all_training_results)
        save_experiment_results(all_training_results, all_eval_results)

        print("\n=== Experiment 1 Performance Summary Table ===")
        print(
            f"{'L Dim':<8} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Epochs':<10} {'Time(s)':<12}"
        )
        print("-" * 65)

        for train_result, eval_result in zip(all_training_results, all_eval_results):
            l_name = train_result["l_dim_name"]
            r2 = eval_result["r2"]
            rmse = eval_result["rmse"]
            mae = eval_result["mae"]
            epochs = train_result["epochs_trained"]
            time_s = train_result["training_time"]

            print(
                f"{l_name:<8} {r2:<10.6f} {rmse:<10.6f} {mae:<10.6f} {epochs:<10} {time_s:<12.2f}"
            )

        print("\n✅ Experiment 1: Single dimension perturbation experiment completed!")
        print(f"   Successfully trained {len(all_training_results)}/5 networks")
        print("   Result files:")
        print("   - Training curves: results/experiment1_training_curves.png")
        print("   - Experiment results: results/experiment1_results.json")
        print("   - Trained models: models/experiment1_*_network.pth")
    else:
        print("\n❌ Experiment 1 failed: No networks trained successfully")


if __name__ == "__main__":
    main()
