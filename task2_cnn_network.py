import os

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
    """PyTorch Dataset class for CNN-based complex impedance regression

    Handles complex impedance data formatted for 1D CNN processing with
    separate real and imaginary channels.
    """

    def __init__(self, z_data, l_data, transform=None):
        """Initialize dataset with impedance and inductance data

        Args:
            z_data: Complex impedance data (N_samples, 2*F_num)
                   Format: [real_0, imag_0, real_1, imag_1, ..., real_1499, imag_1499]
            l_data: Inductance target values (N_samples, N_coils) in microhenries
            transform: Optional data transformation (e.g., normalization)
        """
        # Reshape data for CNN: convert from interleaved [real, imag, real, imag, ...]
        # to separate channels [batch, 2, F_num] where channel 0=real, channel 1=imag
        batch_size, total_features = z_data.shape
        f_num = total_features // 2

        # Reshape from [real_0, imag_0, real_1, imag_1, ...] to [real_array, imag_array]
        z_reshaped = z_data.reshape(batch_size, f_num, 2)  # Shape: (batch, F_num, 2)
        z_channels = z_reshaped.transpose(0, 2, 1)  # Shape: (batch, 2, F_num)

        self.z_data = torch.FloatTensor(z_channels)  # Shape: (batch, 2, F_num)
        self.l_data = torch.FloatTensor(l_data)
        self.transform = transform

    def __len__(self):
        return len(self.z_data)

    def __getitem__(self, idx):
        z_sample = self.z_data[idx]  # Shape: (2, F_num)
        l_sample = self.l_data[idx]

        if self.transform:
            z_sample = self.transform(z_sample)

        return z_sample, l_sample


class CNNImpedanceNetwork(nn.Module):
    """1D Convolutional Neural Network (CNN) for complex impedance regression

    CNN architecture matching specification requirements:
    - Input: [batch, 2, 1500] where channels are real and imaginary parts
    - 3 Conv1D layers: 32→64→128 channels, kernel=5, stride=1, padding=2
    - MaxPool1D after each convolution (kernel=2)
    - Fully connected layers: 256→128→5
    - Output: 5 inductance parameters (L1-L5)
    """

    def __init__(self, f_num=1500, output_size=5):
        """Initialize CNN architecture for complex impedance processing

        Args:
            f_num: Number of frequency points (default: 1500)
            output_size: Number of inductance parameters to predict (default: 5 for L1-L5)
        """
        super(CNNImpedanceNetwork, self).__init__()

        self.f_num = f_num

        # Convolutional layers as per specification
        self.conv_layers = nn.Sequential(
            # Conv Layer 1: 2→32 channels, kernel=5, stride=1, padding=2
            nn.Conv1d(
                in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduces length by factor of 2
            # Conv Layer 2: 32→64 channels, kernel=5, stride=1, padding=2
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduces length by factor of 2
            # Conv Layer 3: 64→128 channels, kernel=5, stride=1, padding=2
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Reduces length by factor of 2
        )

        # Calculate flattened size after convolutions and pooling
        # After 3 pooling layers: f_num / (2^3) = f_num / 8 ≈ 187 for f_num=1500
        conv_output_length = f_num // 8
        flattened_size = 128 * conv_output_length

        # Fully connected layers as per specification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            # FC Layer 1: flattened_size → 256
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            # FC Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Output layer: 128 → 5 (linear activation for regression)
            nn.Linear(128, output_size),
        )

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through CNN

        Args:
            x: Input tensor with shape (batch_size, 2, f_num)
               Channel 0: Real parts, Channel 1: Imaginary parts

        Returns:
            Output tensor with shape (batch_size, 5) for L1-L5 predictions
        """
        # Pass through convolutional layers
        x = self.conv_layers(x)  # Shape: (batch, 128, f_num//8)

        # Pass through fully connected layers
        x = self.fc_layers(x)  # Shape: (batch, 5)

        return x


class CNNNetworkTrainer:
    """Training class for CNN-based impedance neural network"""

    def __init__(self, model, device="cpu", learning_rate=0.001):
        """Initialize CNN trainer

        Args:
            model: CNNImpedanceNetwork instance
            device: Training device ('cpu' or 'cuda')
            learning_rate: Learning rate for Adam optimizer (default: 1e-3 as per spec)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()  # MSE loss for regression as per specification
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Learning rate scheduler with patience-based reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train CNN for one epoch"""
        self.model.train()
        total_loss = 0.0

        for z_batch, l_batch in train_loader:
            z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(z_batch)
            loss = self.criterion(outputs, l_batch)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        """Validate CNN for one epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for z_batch, l_batch in val_loader:
                z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)
                outputs = self.model(z_batch)
                loss = self.criterion(outputs, l_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self, train_loader, val_loader, epochs=200, save_path="models/cnn_network.pth"
    ):
        """Full CNN training loop with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of training epochs (default: 200 as per spec)
            save_path: Path to save trained model
        """
        print(f"Training CNN for complex impedance regression - {epochs} epochs...")

        # Create models directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Early stopping parameters
        patience = 20
        patience_counter = 0

        for epoch in tqdm(range(epochs), desc="CNN Training Progress"):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Save best model and check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
                break

        # Save best model with training history
        torch.save(
            {
                "model_state_dict": self.best_model_state,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "epochs_trained": len(self.train_losses),
            },
            save_path,
        )

        print("CNN training completed!")
        print(f"  - Best validation loss: {self.best_val_loss:.6f}")
        print(f"  - Epochs trained: {len(self.train_losses)}")
        print(f"  - Model saved: {save_path}")

    def evaluate(self, test_loader, model_path="models/cnn_network.pth"):
        """Evaluate trained CNN on test set"""
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for z_batch, l_batch in test_loader:
                z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)
                outputs = self.model(z_batch)

                predictions.append(outputs.cpu().numpy())
                targets.append(l_batch.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # Calculate regression metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)

        print("\n=== CNN Network Evaluation Results ===")
        print(f"Test MSE:  {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")
        print(f"Test R²:   {r2:.6f}")

        # Per-coil analysis
        print("\nPer-coil performance:")
        coil_names = ["L1", "L2", "L3", "L4", "L5"]
        for i, name in enumerate(coil_names):
            coil_mse = mean_squared_error(targets[:, i], predictions[:, i])
            coil_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            coil_r2 = r2_score(targets[:, i], predictions[:, i])
            print(
                f"  {name} - MSE: {coil_mse:.6f}, MAE: {coil_mae:.6f}, R²: {coil_r2:.6f}"
            )

        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def plot_training_curves(self, save_path="results/cnn_network_training.png"):
        """Plot CNN training and validation loss curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(12, 5))

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss", color="blue", alpha=0.7)
        plt.plot(self.val_losses, label="Validation Loss", color="red", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("CNN Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Learning rate evolution (if available)
        plt.subplot(1, 2, 2)
        if hasattr(self.scheduler, "_last_lr"):
            lr_history = [self.optimizer.param_groups[0]["lr"]] * len(self.train_losses)
            plt.plot(lr_history, label="Learning Rate", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale("log")
        else:
            plt.text(
                0.5,
                0.5,
                "Learning Rate\nHistory Not Available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Learning Rate Schedule")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Training curves saved to: {save_path}")


def load_and_prepare_cnn_data(
    csv_path="mixed_z_dataset_2.csv", test_size=0.2, val_size=0.2, random_state=42
):
    """Load and prepare complex impedance dataset for CNN training

    Args:
        csv_path: Path to CSV file with complex impedance data (default: L3 dataset)
        test_size: Fraction of data for testing (default: 0.2)
        val_size: Fraction of remaining data for validation (default: 0.2)
        random_state: Random seed for reproducible splits (default: 42)

    Returns:
        dict: Training, validation, test splits and data scalers
    """
    print(f"Loading complex impedance data for CNN training: {csv_path}")

    # Load dataset using Polars
    df = pl.read_csv(csv_path)

    # Extract complex impedance features and inductance targets
    z_columns = [
        col
        for col in df.columns
        if col.startswith("z_") and ("real" in col or "imag" in col)
    ]
    l_columns = [col for col in df.columns if col.startswith("L") and len(col) == 2]

    # Convert to numpy arrays
    z_data = df.select(z_columns).to_numpy()  # Shape: (N_samples, 2*F_num)
    l_data = df.select(l_columns).to_numpy()  # Shape: (N_samples, N_coils)

    print("CNN dataset loaded successfully:")
    print(f"  - Complex impedance shape: {z_data.shape}")
    print("  - Expected: (1000, 3000) for 1500 freq points × 2")
    print(f"  - Inductance targets shape: {l_data.shape}")
    print(f"  - Z magnitude range: [{z_data.min():.2f}, {z_data.max():.2f}] Ohms")
    print(f"  - L parameter range: [{l_data.min():.2f}, {l_data.max():.2f}] μH")

    # Split dataset: 80% train, 10% validation, 10% test
    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=test_size, random_state=random_state
    )

    z_train, z_val, l_train, l_val = train_test_split(
        z_temp, l_temp, test_size=val_size / (1 - test_size), random_state=random_state
    )

    # Normalize complex impedance features
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)

    # Normalize inductance targets
    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)

    print("CNN data split completed:")
    print(
        f"  - Training samples: {len(z_train)} ({len(z_train) / len(z_data) * 100:.1f}%)"
    )
    print(
        f"  - Validation samples: {len(z_val)} ({len(z_val) / len(z_data) * 100:.1f}%)"
    )
    print(f"  - Test samples: {len(z_test)} ({len(z_test) / len(z_data) * 100:.1f}%)")

    return {
        "train": (z_train, l_train),
        "val": (z_val, l_val),
        "test": (z_test, l_test),
        "scalers": {"z_scaler": z_scaler, "l_scaler": l_scaler},
    }


def main():
    """Main function to train CNN for complex impedance regression"""
    print("=== CNN (Convolutional Neural Network) Training ===")
    print("Training 1D CNN on complex impedance spectra for L parameter prediction")

    # Set computational device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare dataset (using L3 dataset as specified for model comparison)
    try:
        data = load_and_prepare_cnn_data(
            "mixed_z_dataset_2.csv"
        )  # L3 variation dataset
    except FileNotFoundError:
        print("Error: mixed_z_dataset_2.csv not found.")
        print(
            "Please run: uv run task1_improved_dataset.py first to generate datasets."
        )
        return

    # Create CNN-specific datasets and data loaders
    train_dataset = ComplexImpedanceDataset(data["train"][0], data["train"][1])
    val_dataset = ComplexImpedanceDataset(data["val"][0], data["val"][1])
    test_dataset = ComplexImpedanceDataset(data["test"][0], data["test"][1])

    # Use batch size of 64 as per specification
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("\n" + "=" * 60)
    print("CNN Architecture Details")
    print("=" * 60)

    # Create CNN model matching specification
    model = CNNImpedanceNetwork(f_num=1500, output_size=5)
    print("CNN Architecture:")
    print("  - Input: [batch, 2, 1500] (2 channels: real/imaginary parts)")
    print("  - Conv1D layers: [2→32→64→128] channels, kernel=5, MaxPool1D(2)")
    print("  - FC layers: [flattened→256→128→5] with ReLU activation")
    print("  - Output: 5 neurons (L1-L5 inductance parameters)")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer with Adam optimizer
    trainer = CNNNetworkTrainer(model, device=device, learning_rate=0.001)

    # Train CNN model
    trainer.train(
        train_loader, val_loader, epochs=200, save_path="models/cnn_network.pth"
    )
    trainer.plot_training_curves("results/cnn_training.png")

    # Evaluate on test set
    results = trainer.evaluate(test_loader, "models/cnn_network.pth")

    # Performance summary
    print("\n" + "=" * 60)
    print("CNN Training Summary")
    print("=" * 60)
    print("✅ CNN training completed successfully!")
    print("  - Architecture: 1D CNN with 3 conv layers + 2 FC layers")
    print(f"  - Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"  - Test R²: {results['r2']:.6f}")
    print(f"  - Test RMSE: {results['rmse']:.6f}")
    print("  - Model saved: models/cnn_network.pth")
    print("  - Training curves: results/cnn_training.png")


if __name__ == "__main__":
    main()
