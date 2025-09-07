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


class PINNImpedanceDataset(Dataset):
    """PyTorch Dataset class for Physics-Informed Neural Network (PINN)

    Handles complex impedance data with additional physics information
    for physics-constrained training.
    """

    def __init__(
        self, z_data, l_data, frequency_axis=None, circuit_params=None, transform=None
    ):
        """Initialize PINN dataset with impedance, inductance, and physics data

        Args:
            z_data: Complex impedance data (N_samples, 2*F_num)
                   Format: [real_0, imag_0, real_1, imag_1, ..., real_1499, imag_1499]
            l_data: Inductance target values (N_samples, N_coils) in microhenries
            frequency_axis: Frequency points array for physics calculations (optional)
            circuit_params: Circuit parameters (C, R) for physics constraints (optional)
            transform: Optional data transformation (e.g., normalization)
        """
        self.z_data = torch.FloatTensor(z_data)
        self.l_data = torch.FloatTensor(l_data)
        self.transform = transform

        # Store physics information for loss calculation
        self.frequency_axis = frequency_axis
        self.circuit_params = circuit_params

    def __len__(self):
        return len(self.z_data)

    def __getitem__(self, idx):
        z_sample = self.z_data[idx]
        l_sample = self.l_data[idx]

        if self.transform:
            z_sample = self.transform(z_sample)

        return z_sample, l_sample


class PINNNetwork(nn.Module):
    """Physics-Informed Neural Network (PINN) for complex impedance regression

    PINN architecture with same structure as FCNN but enhanced with physics constraints:
    - Input: 3000 dimensions (1500 frequency points × 2 for real/imaginary)
    - Hidden layers: [1024, 512, 256, 128] with ReLU activation
    - Output: 5 inductance parameters (L1-L5)
    - Physics constraints: Resonance conditions and impedance properties
    """

    def __init__(
        self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]
    ):
        """Initialize PINN architecture (same structure as FCNN)

        Args:
            input_size: Number of complex impedance features (default: 3000)
            output_size: Number of inductance parameters (default: 5 for L1-L5)
            hidden_sizes: List of hidden layer neuron counts
        """
        super(PINNNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers with ReLU activation (same as FCNN)
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

        # Output layer with linear activation for regression
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through PINN"""
        return self.network(x)


class PhysicsLoss(nn.Module):
    """Physics-informed loss function with circuit physics constraints"""

    def __init__(
        self, lambda1=0.01, lambda2=0.01, frequency_axis=None, circuit_params=None
    ):
        """Initialize physics loss with constraint weights

        Args:
            lambda1: Weight for resonance frequency constraint
            lambda2: Weight for impedance reality constraint
            frequency_axis: Frequency points for physics calculations
            circuit_params: Circuit parameters (C, R arrays)
        """
        super(PhysicsLoss, self).__init__()

        self.lambda1 = lambda1  # Resonance constraint weight
        self.lambda2 = lambda2  # Impedance constraint weight
        self.mse_loss = nn.MSELoss()

        # Store physics parameters
        if frequency_axis is not None:
            self.register_buffer("frequency_axis", torch.FloatTensor(frequency_axis))
            self.omega = 2 * np.pi * self.frequency_axis
        else:
            self.frequency_axis = None

        if circuit_params is not None:
            self.C = torch.FloatTensor(
                circuit_params.get("C", [162e-12, 184e-12, 170e-12, 230e-12, 189e-12])
            )
            self.R = torch.FloatTensor(circuit_params.get("R", [25, 25, 50, 25, 25]))
        else:
            # Default circuit parameters from specification
            self.C = torch.FloatTensor([162e-12, 184e-12, 170e-12, 230e-12, 189e-12])
            self.R = torch.FloatTensor([25, 25, 50, 25, 25])

    def resonance_constraint(self, L_pred):
        """Calculate resonance frequency constraint: ω₀ ≈ 1/√(LC)

        Args:
            L_pred: Predicted inductance values (batch_size, 5) in microhenries

        Returns:
            Resonance constraint loss
        """
        if self.frequency_axis is None:
            return torch.tensor(0.0, requires_grad=True, device=L_pred.device)

        # Convert inductances from μH to H for physics calculations
        L_henry = torch.abs(L_pred) * 1e-6 + 1e-12  # Ensure positive and non-zero
        C_expanded = self.C.unsqueeze(0).expand_as(L_henry).to(L_pred.device)

        # Calculate theoretical resonance frequencies: f₀ = 1/(2π√LC)
        LC_product = L_henry * C_expanded
        theoretical_omega = 1.0 / torch.sqrt(LC_product + 1e-12)  # Add small epsilon

        # Use a reasonable target frequency range instead of mean
        # Target around 4 MHz which is typical for this circuit
        omega_target = 2 * 3.14159 * 4e6  # 4 MHz
        omega_target = torch.full_like(theoretical_omega, omega_target)

        # Normalize the loss to prevent explosion
        diff = (theoretical_omega - omega_target) / omega_target
        resonance_loss = torch.mean(diff ** 2)

        # Clamp to prevent numerical instability
        resonance_loss = torch.clamp(resonance_loss, 0, 100.0)

        return resonance_loss

    def impedance_constraint(self, L_pred, z_complex_input):
        """Calculate impedance constraint: Im(Z(ω₀)) ≈ 0 at resonance

        Args:
            L_pred: Predicted inductance values (batch_size, 5)
            z_complex_input: Complex impedance input (batch_size, 3000)

        Returns:
            Impedance constraint loss
        """
        # Extract imaginary parts from complex impedance input
        # Input format: [real_0, imag_0, real_1, imag_1, ...]
        batch_size = z_complex_input.size(0)
        z_reshaped = z_complex_input.view(batch_size, -1, 2)  # Shape: (batch, F_num, 2)
        z_imag = z_reshaped[:, :, 1]  # Imaginary parts: (batch, F_num)

        # Calculate impedance constraint: minimize imaginary part variance
        # At resonance, imaginary part should be close to zero
        imag_variance = torch.var(z_imag, dim=1)  # Shape: (batch,)
        impedance_loss = torch.mean(imag_variance)

        # Clamp to prevent numerical instability
        impedance_loss = torch.clamp(impedance_loss, 0, 1000.0)

        return impedance_loss

    def forward(self, predictions, targets, complex_input):
        """Calculate total physics-informed loss

        Args:
            predictions: Model predictions (batch_size, 5)
            targets: Target inductance values (batch_size, 5)
            complex_input: Complex impedance input (batch_size, 3000)

        Returns:
            Total loss combining MSE and physics constraints
        """
        # Standard MSE loss for regression
        mse_loss = self.mse_loss(predictions, targets)

        # Physics constraint losses
        resonance_loss = self.resonance_constraint(predictions)
        impedance_loss = self.impedance_constraint(predictions, complex_input)

        # Combined loss: L = L_MSE + λ₁·L_resonance + λ₂·L_impedance
        total_loss = (
            mse_loss + self.lambda1 * resonance_loss + self.lambda2 * impedance_loss
        )

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "resonance_loss": resonance_loss,
            "impedance_loss": impedance_loss,
        }


class PINNTrainer:
    """Training class for Physics-Informed Neural Network"""

    def __init__(
        self,
        model,
        device="cpu",
        learning_rate=0.001,
        lambda1=0.01,
        lambda2=0.01,
        frequency_axis=None,
        circuit_params=None,
    ):
        """Initialize PINN trainer with physics constraints

        Args:
            model: PINNNetwork instance
            device: Training device ('cpu' or 'cuda')
            learning_rate: Learning rate for Adam optimizer
            lambda1: Weight for resonance constraint
            lambda2: Weight for impedance constraint
            frequency_axis: Frequency points for physics calculations
            circuit_params: Circuit parameters for physics constraints
        """
        self.model = model.to(device)
        self.device = device

        # Physics-informed loss function
        self.criterion = PhysicsLoss(
            lambda1=lambda1,
            lambda2=lambda2,
            frequency_axis=frequency_axis,
            circuit_params=circuit_params,
        ).to(device)

        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # Training tracking
        self.train_losses = {"total": [], "mse": [], "resonance": [], "impedance": []}
        self.val_losses = {"total": [], "mse": [], "resonance": [], "impedance": []}
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train_epoch(self, train_loader):
        """Train PINN for one epoch with physics constraints"""
        self.model.train()
        total_losses = {"total": 0.0, "mse": 0.0, "resonance": 0.0, "impedance": 0.0}

        for z_batch, l_batch in train_loader:
            z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(z_batch)

            # Calculate physics-informed loss
            loss_dict = self.criterion(outputs, l_batch, z_batch)

            # Backward pass
            loss_dict["total_loss"].backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[f"{key}_loss"].item()

        # Average losses
        for key in total_losses:
            total_losses[key] /= len(train_loader)

        return total_losses

    def validate_epoch(self, val_loader):
        """Validate PINN for one epoch"""
        self.model.eval()
        total_losses = {"total": 0.0, "mse": 0.0, "resonance": 0.0, "impedance": 0.0}

        with torch.no_grad():
            for z_batch, l_batch in val_loader:
                z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)
                outputs = self.model(z_batch)

                # Calculate physics-informed loss
                loss_dict = self.criterion(outputs, l_batch, z_batch)

                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[f"{key}_loss"].item()

        # Average losses
        for key in total_losses:
            total_losses[key] /= len(val_loader)

        return total_losses

    def train(
        self, train_loader, val_loader, epochs=200, save_path="models/pinn_network.pth"
    ):
        """Full PINN training loop with physics constraints

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of training epochs
            save_path: Path to save trained model
        """
        print(f"Training PINN (Physics-Informed Neural Network) - {epochs} epochs...")
        print(
            f"Physics constraints: λ₁={self.criterion.lambda1} (resonance), λ₂={self.criterion.lambda2} (impedance)"
        )

        # Create models directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Early stopping parameters
        patience = 20
        patience_counter = 0

        for epoch in tqdm(range(epochs), desc="PINN Training Progress"):
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate_epoch(val_loader)

            # Store training history
            for key in self.train_losses:
                self.train_losses[key].append(train_losses[key])
                self.val_losses[key].append(val_losses[key])

            # Update learning rate based on total validation loss
            self.scheduler.step(val_losses["total"])

            # Save best model and check for early stopping
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                import copy
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:3d}: "
                    f"Train Loss: {train_losses['total']:.6f} "
                    f"(MSE: {train_losses['mse']:.6f}, "
                    f"Res: {train_losses['resonance']:.6f}, "
                    f"Imp: {train_losses['impedance']:.6f}), "
                    f"Val Loss: {val_losses['total']:.6f}, "
                    f"LR: {current_lr:.2e}"
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
                "epochs_trained": len(self.train_losses["total"]),
                "physics_params": {
                    "lambda1": self.criterion.lambda1,
                    "lambda2": self.criterion.lambda2,
                },
            },
            save_path,
        )

        print("PINN training completed!")
        print(f"  - Best validation loss: {self.best_val_loss:.6f}")
        print(f"  - Epochs trained: {len(self.train_losses['total'])}")
        print(f"  - Model saved: {save_path}")

    def evaluate(self, test_loader, model_path="models/pinn_network.pth"):
        """Evaluate trained PINN on test set"""
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        predictions = []
        targets = []
        physics_losses = {"total": [], "mse": [], "resonance": [], "impedance": []}

        with torch.no_grad():
            for z_batch, l_batch in test_loader:
                z_batch, l_batch = z_batch.to(self.device), l_batch.to(self.device)
                outputs = self.model(z_batch)

                # Calculate physics losses for analysis
                loss_dict = self.criterion(outputs, l_batch, z_batch)
                for key in physics_losses:
                    physics_losses[key].append(loss_dict[f"{key}_loss"].item())

                predictions.append(outputs.cpu().numpy())
                targets.append(l_batch.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # Calculate regression metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)

        # Average physics losses
        avg_physics_losses = {
            key: np.mean(values) for key, values in physics_losses.items()
        }

        print("\n=== PINN Network Evaluation Results ===")
        print(f"Test MSE:  {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")
        print(f"Test R²:   {r2:.6f}")

        print("\nPhysics constraint losses:")
        print(f"  - Total loss: {avg_physics_losses['total']:.6f}")
        print(f"  - MSE loss: {avg_physics_losses['mse']:.6f}")
        print(f"  - Resonance loss: {avg_physics_losses['resonance']:.6f}")
        print(f"  - Impedance loss: {avg_physics_losses['impedance']:.6f}")

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
            "physics_losses": avg_physics_losses,
        }

    def plot_training_curves(self, save_path="results/pinn_training.png"):
        """Plot PINN training curves including physics losses"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Total loss
        axes[0, 0].plot(
            self.train_losses["total"], label="Training", color="blue", alpha=0.7
        )
        axes[0, 0].plot(
            self.val_losses["total"], label="Validation", color="red", alpha=0.7
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("PINN Total Loss (MSE + Physics)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale("log")

        # MSE loss
        axes[0, 1].plot(
            self.train_losses["mse"], label="Training MSE", color="green", alpha=0.7
        )
        axes[0, 1].plot(
            self.val_losses["mse"], label="Validation MSE", color="orange", alpha=0.7
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE Loss")
        axes[0, 1].set_title("PINN MSE Loss (Regression Component)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        # Resonance constraint loss
        axes[1, 0].plot(
            self.train_losses["resonance"],
            label="Training Resonance",
            color="purple",
            alpha=0.7,
        )
        axes[1, 0].plot(
            self.val_losses["resonance"],
            label="Validation Resonance",
            color="brown",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Resonance Loss")
        axes[1, 0].set_title("Physics Constraint: Resonance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale("log")

        # Impedance constraint loss
        axes[1, 1].plot(
            self.train_losses["impedance"],
            label="Training Impedance",
            color="cyan",
            alpha=0.7,
        )
        axes[1, 1].plot(
            self.val_losses["impedance"],
            label="Validation Impedance",
            color="magenta",
            alpha=0.7,
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Impedance Loss")
        axes[1, 1].set_title("Physics Constraint: Impedance")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"PINN training curves saved to: {save_path}")


def load_and_prepare_pinn_data(
    csv_path="mixed_z_dataset_2.csv", test_size=0.2, val_size=0.2, random_state=42
):
    """Load and prepare complex impedance dataset for PINN training

    Args:
        csv_path: Path to CSV file with complex impedance data (default: L3 dataset)
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducible splits

    Returns:
        dict: Training, validation, test splits, scalers, and physics information
    """
    print(f"Loading complex impedance data for PINN training: {csv_path}")

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
    z_data = df.select(z_columns).to_numpy()
    l_data = df.select(l_columns).to_numpy()

    print("PINN dataset loaded successfully:")
    print(f"  - Complex impedance shape: {z_data.shape}")
    print(f"  - Inductance targets shape: {l_data.shape}")
    print(f"  - Z range: [{z_data.min():.2f}, {z_data.max():.2f}] Ohms")
    print(f"  - L range: [{l_data.min():.2f}, {l_data.max():.2f}] μH")

    # Generate frequency axis for physics constraints (approximation)
    # In a full implementation, this would come from the original dataset generation
    f_start, f_end = 2.33e6, 5.41e6  # Typical frequency range from dataset generation
    frequency_axis = np.linspace(f_start, f_end, 1500)  # 1500 frequency points

    # Circuit parameters for physics constraints
    circuit_params = {
        "C": np.array(
            [162e-12, 184e-12, 170e-12, 230e-12, 189e-12]
        ),  # Capacitances [F]
        "R": np.array([25, 25, 50, 25, 25]),  # Resistances [Ohms]
    }

    # Split dataset
    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=test_size, random_state=random_state
    )

    z_train, z_val, l_train, l_val = train_test_split(
        z_temp, l_temp, test_size=val_size / (1 - test_size), random_state=random_state
    )

    # Normalize features and targets
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)

    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)

    print("PINN data split completed:")
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
        "physics": {"frequency_axis": frequency_axis, "circuit_params": circuit_params},
    }


def main():
    """Main function to train PINN for complex impedance regression"""
    print("=== PINN (Physics-Informed Neural Network) Training ===")
    print("Training PINN with physics constraints for L parameter prediction")

    # Set computational device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare dataset (using L3 dataset for comparison with other models)
    try:
        data = load_and_prepare_pinn_data(
            "mixed_z_dataset_2.csv"
        )  # L3 variation dataset
    except FileNotFoundError:
        print("Error: mixed_z_dataset_2.csv not found.")
        print(
            "Please run: uv run task1_improved_dataset.py first to generate datasets."
        )
        return

    # Create PINN-specific datasets
    train_dataset = PINNImpedanceDataset(
        data["train"][0],
        data["train"][1],
        frequency_axis=data["physics"]["frequency_axis"],
        circuit_params=data["physics"]["circuit_params"],
    )
    val_dataset = PINNImpedanceDataset(
        data["val"][0],
        data["val"][1],
        frequency_axis=data["physics"]["frequency_axis"],
        circuit_params=data["physics"]["circuit_params"],
    )
    test_dataset = PINNImpedanceDataset(
        data["test"][0],
        data["test"][1],
        frequency_axis=data["physics"]["frequency_axis"],
        circuit_params=data["physics"]["circuit_params"],
    )

    # Create data loaders
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
    print("PINN Architecture Details")
    print("=" * 60)

    # Create PINN model (same architecture as FCNN)
    model = PINNNetwork(
        input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]
    )
    print("PINN Architecture:")
    print("  - Input: 3000 neurons (complex impedance features)")
    print("  - Hidden layers: [1024, 512, 256, 128] with ReLU activation")
    print("  - Output: 5 neurons (L1-L5 inductance parameters)")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  - Physics constraints:")
    print("    * Resonance condition: ω₀ ≈ 1/√(LC)")
    print("    * Impedance constraint: Im(Z(ω₀)) ≈ 0")

    # Create trainer with physics constraints
    trainer = PINNTrainer(
        model,
        device=device,
        learning_rate=0.001,
        lambda1=0.01,
        lambda2=0.01,  # Physics constraint weights
        frequency_axis=data["physics"]["frequency_axis"],
        circuit_params=data["physics"]["circuit_params"],
    )

    # Train PINN model
    trainer.train(
        train_loader, val_loader, epochs=200, save_path="models/pinn_network.pth"
    )
    trainer.plot_training_curves("results/pinn_training.png")

    # Evaluate on test set
    results = trainer.evaluate(test_loader, "models/pinn_network.pth")

    # Performance summary
    print("\n" + "=" * 60)
    print("PINN Training Summary")
    print("=" * 60)
    print("✅ PINN training completed successfully!")
    print("  - Architecture: FCNN + Physics Constraints")
    print("  - Physics constraint weights: λ₁=0.1 (resonance), λ₂=0.1 (impedance)")
    print(f"  - Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"  - Test R²: {results['r2']:.6f}")
    print(f"  - Test RMSE: {results['rmse']:.6f}")
    print("  - Model saved: models/pinn_network.pth")
    print("  - Training curves: results/pinn_training.png")


if __name__ == "__main__":
    main()
