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


class MixedZImpedanceDataset(Dataset):
    """PyTorch Dataset class for complex impedance regression

    Handles complex impedance data with interleaved real/imaginary parts
    and corresponding inductance parameter targets.
    """

    def __init__(self, z_data, l_data, transform=None):
        """Initialize dataset with impedance and inductance data

        Args:
            z_data: Complex impedance data (N_samples, 2*F_num)
                   Format: [real_0, imag_0, real_1, imag_1, ..., real_1499, imag_1499]
            l_data: Inductance target values (N_samples, N_coils) in microhenries
            transform: Optional data transformation (e.g., normalization)
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


class MixedZValueNetwork(nn.Module):
    """Fully Connected Neural Network (FCNN) for complex impedance regression

    Standard feed-forward network architecture matching specification requirements:
    - Input: 3000 dimensions (1500 frequency points × 2 for real/imaginary)
    - Hidden layers: [1024, 512, 256, 128] with ReLU activation
    - Output: 5 inductance parameters (L1-L5)
    """

    def __init__(
        self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]
    ):
        """Initialize FCNN architecture for complex impedance processing

        Args:
            input_size: Number of complex impedance features (default: 3000 for 1500 freq points)
            output_size: Number of inductance parameters to predict (default: 5 for L1-L5)
            hidden_sizes: List of hidden layer neuron counts (default: [1024, 512, 256, 128])
        """
        super(MixedZValueNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers with ReLU activation as per specification
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),  # ReLU activation as specified
                    nn.BatchNorm1d(
                        hidden_size
                    ),  # Batch normalization for training stability
                    nn.Dropout(0.3),  # Dropout for regularization
                ]
            )
            prev_size = hidden_size

        # Output layer with linear activation (no activation function)
        layers.append(nn.Linear(prev_size, output_size))  # Linear output for regression

        self.network = nn.Sequential(*layers)

        # Initialize network weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class ComplexFeatureExtractor(nn.Module):
    """Alternative network with explicit complex feature extraction"""

    def __init__(self, f_num=1000, output_size=5):
        """
        Args:
            f_num: Number of frequency points
            output_size: Number of inductance values to predict (N_coils)
        """
        super(ComplexFeatureExtractor, self).__init__()
        self.f_num = f_num
        # Complex feature extraction layers
        self.magnitude_layer = nn.Sequential(
            nn.Linear(f_num, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2)
        )
        self.phase_layer = nn.Sequential(
            nn.Linear(f_num, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2)
        )
        # Combined processing
        self.combined_layers = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass with complex feature extraction
        Args:
            x: Input tensor with shape (batch_size, 2*f_num)
               Format: [real1, imag1, real2, imag2, ..., real_f_num, imag_f_num]
        """
        batch_size = x.size(0)
        # Reshape to separate real and imaginary parts
        # x shape: (batch_size, 2*f_num) -> (batch_size, f_num, 2)
        x_reshaped = x.view(batch_size, self.f_num, 2)
        real_parts = x_reshaped[:, :, 0]  # Shape: (batch_size, f_num)
        imag_parts = x_reshaped[:, :, 1]  # Shape: (batch_size, f_num)
        # Calculate magnitude and phase
        magnitude = torch.sqrt(real_parts**2 + imag_parts**2)
        phase = torch.atan2(imag_parts, real_parts)
        # Process through separate layers
        mag_features = self.magnitude_layer(magnitude)
        phase_features = self.phase_layer(phase)
        # Combine features
        combined = torch.cat([mag_features, phase_features], dim=1)
        # Final processing
        output = self.combined_layers(combined)
        return output


class MixedZNetworkTrainer:
    """Training class for Mixed Z-value neural network"""

    def __init__(self, model, device="cpu", learning_rate=0.001):
        """
        Args:
            model: MixedZValueNetwork instance
            device: Training device ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for z_batch, l_batch in train_loader:
            z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(z_batch)
            loss = self.criterion(outputs, l_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
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
        self,
        train_loader,
        val_loader,
        epochs=100,
        save_path="models/mixed_z_network.pth",
    ):
        """Full training loop"""
        print(f"Training Mixed Z-value network for {epochs} epochs...")

        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        # Save best model
        torch.save(
            {
                "model_state_dict": self.best_model_state,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
            },
            save_path,
        )

        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        print(f"Model saved to: {save_path}")

    def evaluate(self, test_loader, model_path="models/mixed_z_network.pth"):
        """Evaluate trained model on test set"""
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

        # Calculate metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        print("\n=== Mixed Z-Value Network Evaluation ===")
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test R²:  {r2:.6f}")

        # Per-coil analysis
        coil_names = ["L1", "L2", "L3", "L4", "L5"]
        for i, name in enumerate(coil_names):
            coil_mse = mean_squared_error(targets[:, i], predictions[:, i])
            coil_r2 = r2_score(targets[:, i], predictions[:, i])
            print(f"{name} - MSE: {coil_mse:.6f}, R²: {coil_r2:.6f}")

        return {
            "predictions": predictions,
            "targets": targets,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }

    def plot_training_curves(self, save_path="results/mixed_z_network_training.png"):
        """Plot training and validation loss curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", color="blue")
        plt.plot(self.val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Mixed Z-Value Network Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Training curves saved to: {save_path}")


def load_and_prepare_mixed_data(
    csv_path="mixed_z_dataset_0.csv", test_size=0.2, val_size=0.2, random_state=42
):
    """Load and prepare complex impedance dataset for neural network training

    Args:
        csv_path: Path to CSV file with complex impedance data
        test_size: Fraction of data for testing (default: 0.2)
        val_size: Fraction of remaining data for validation (default: 0.2)
        random_state: Random seed for reproducible splits (default: 42)

    Returns:
        dict: Training, validation, test splits and data scalers
    """
    print(f"Loading complex impedance data from {csv_path}...")

    # Load dataset using Polars for better performance
    df = pl.read_csv(csv_path)

    # Extract complex impedance features and inductance targets
    z_columns = [
        col
        for col in df.columns
        if col.startswith("z_") and ("real" in col or "imag" in col)
    ]
    l_columns = [
        col for col in df.columns if col.startswith("L") and len(col) == 2
    ]  # L1, L2, L3, L4, L5

    # Convert to numpy arrays for sklearn compatibility
    z_data = df.select(z_columns).to_numpy()  # Shape: (N_samples, 2*F_num)
    l_data = df.select(l_columns).to_numpy()  # Shape: (N_samples, N_coils)

    print("Dataset loaded successfully:")
    print(f"  - Complex impedance shape: {z_data.shape}")
    print(f"  - Inductance targets shape: {l_data.shape}")
    print(f"  - Z magnitude range: [{z_data.min():.2f}, {z_data.max():.2f}] Ohms")
    print(f"  - L parameter range: [{l_data.min():.2f}, {l_data.max():.2f}] μH")

    # Split dataset: 80% train, 10% validation, 10% test (as per specification)
    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=test_size, random_state=random_state, stratify=None
    )

    z_train, z_val, l_train, l_val = train_test_split(
        z_temp,
        l_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=None,
    )

    # Normalize complex impedance features using StandardScaler
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)

    # Normalize inductance targets for better training stability
    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)

    print("Data split completed:")
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
    """Main function to train complex impedance neural network models

    Trains both standard FCNN and complex feature extractor architectures
    on complex impedance data to predict inductance parameters.
    """
    print("=== Complex Impedance Neural Network Training ===")
    print(
        "Training FCNN models on complex impedance spectra for L parameter prediction"
    )

    # Set computational device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare complex impedance dataset
    try:
        data = load_and_prepare_mixed_data(
            "mixed_z_dataset_0.csv"
        )  # Use L1 variation dataset
    except FileNotFoundError:
        print("Error: mixed_z_dataset_0.csv not found.")
        print(
            "Please run: uv run task1_improved_dataset.py first to generate datasets."
        )
        return

    # Create PyTorch datasets and data loaders
    train_dataset = MixedZImpedanceDataset(data["train"][0], data["train"][1])
    val_dataset = MixedZImpedanceDataset(data["val"][0], data["val"][1])
    test_dataset = MixedZImpedanceDataset(data["test"][0], data["test"][1])

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

    # Get input dimensions and create FCNN model
    input_size = data["train"][0].shape[1]  # Should be 3000 (1500 freq points × 2)
    print("\nDataset information:")
    print(f"  - Input features: {input_size} (complex impedance values)")
    print("  - Expected: 3000 (1500 frequency points × 2 for real/imaginary)")

    print("\n" + "=" * 60)
    print("Training Standard FCNN (Fully Connected Neural Network)")
    print("=" * 60)

    # Create standard FCNN model matching specification
    model1 = MixedZValueNetwork(
        input_size=input_size, output_size=5, hidden_sizes=[1024, 512, 256, 128]
    )
    print("FCNN Architecture:")
    print(f"  - Input layer: {input_size} neurons")
    print("  - Hidden layers: [1024, 512, 256, 128] neurons with ReLU activation")
    print("  - Output layer: 5 neurons (L1-L5 inductance parameters)")
    print(f"  - Total parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Create trainer with Adam optimizer (lr=1e-3 as per specification)
    trainer1 = MixedZNetworkTrainer(model1, device=device, learning_rate=0.001)
    trainer1.train(
        train_loader, val_loader, epochs=200, save_path="models/mixed_z_fcnn.pth"
    )  # 200 epochs as specified
    trainer1.plot_training_curves("results/mixed_z_fcnn_training.png")
    results1 = trainer1.evaluate(test_loader, "models/mixed_z_fcnn.pth")

    print("\n" + "=" * 60)
    print("Testing Complex Feature Extractor Network:")

    # Test alternative complex feature extraction network
    f_num = input_size // 2  # Number of frequency points
    model2 = ComplexFeatureExtractor(f_num=f_num)
    print(f"Model 2 - F_num: {f_num}")
    print(f"Total parameters: {sum(p.numel() for p in model2.parameters()):,}")

    trainer2 = MixedZNetworkTrainer(model2, device=device, learning_rate=0.0005)
    trainer2.train(
        train_loader,
        val_loader,
        epochs=100,
        save_path="models/complex_extractor_network.pth",
    )
    trainer2.plot_training_curves("results/complex_extractor_training.png")
    results2 = trainer2.evaluate(test_loader, "models/complex_extractor_network.pth")

    # Compare results
    print("\n" + "=" * 60)
    print("=== MODEL COMPARISON ===")
    print(
        f"Standard Mixed Z-Network - R²: {results1['r2']:.6f}, MSE: {results1['mse']:.6f}"
    )
    print(
        f"Complex Feature Network - R²: {results2['r2']:.6f}, MSE: {results2['mse']:.6f}"
    )

    if results2["r2"] > results1["r2"]:
        print("✅ Complex Feature Extractor performs better!")
    else:
        print("✅ Standard Mixed Z-Network performs better!")

    print("\n✅ Mixed Z-Value network training completed successfully!")


if __name__ == "__main__":
    main()
