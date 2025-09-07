import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from tqdm import tqdm


class ZImpedanceDataset(Dataset):
    """Dataset class for Z-magnitude impedance regression"""
    
    def __init__(self, z_data, l_data, transform=None):
        """
        Args:
            z_data: Impedance magnitude data (N_samples, F_num)
            l_data: Inductance target values (N_samples, N_coils)
            transform: Optional transform to apply to data
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


class ZValueNetwork(nn.Module):
    """Neural network for Z-magnitude to inductance regression"""
    
    def __init__(self, input_size=1000, output_size=5, hidden_sizes=[512, 256, 128, 64]):
        """
        Args:
            input_size: Number of impedance magnitude features (F_num)
            output_size: Number of inductance values to predict (N_coils)
            hidden_sizes: List of hidden layer sizes
        """
        super(ZValueNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
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


class ZNetworkTrainer:
    """Training class for Z-value neural network"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Args:
            model: ZValueNetwork instance
            device: Training device ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
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
    
    def train(self, train_loader, val_loader, epochs=100, save_path='models/z_network.pth'):
        """Full training loop"""
        print(f"Training Z-value network for {epochs} epochs...")
        
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
                print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        torch.save({
            'model_state_dict': self.best_model_state,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
        print(f"Model saved to: {save_path}")
    
    def evaluate(self, test_loader, model_path='models/z_network.pth'):
        """Evaluate trained model on test set"""
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        
        print("\n=== Z-Value Network Evaluation ===")
        print(f"Test MSE: {mse:.6f}")
        print(f"Test MAE: {mae:.6f}")
        print(f"Test R²:  {r2:.6f}")
        
        # Per-coil analysis
        coil_names = ['L1', 'L2', 'L3', 'L4', 'L5']
        for i, name in enumerate(coil_names):
            coil_mse = mean_squared_error(targets[:, i], predictions[:, i])
            coil_r2 = r2_score(targets[:, i], predictions[:, i])
            print(f"{name} - MSE: {coil_mse:.6f}, R²: {coil_r2:.6f}")
        
        return {
            'predictions': predictions,
            'targets': targets,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_training_curves(self, save_path='results/z_network_training.png'):
        """Plot training and validation loss curves"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Z-Value Network Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {save_path}")


def load_and_prepare_data(csv_path='z_dataset.csv', test_size=0.2, val_size=0.2, random_state=42):
    """Load and prepare Z-dataset for training"""
    print(f"Loading data from {csv_path}...")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Extract features (impedance magnitudes) and targets (inductances)
    z_columns = [col for col in df.columns if col.startswith('z') and not col.endswith('_real') and not col.endswith('_imag')]
    l_columns = [col for col in df.columns if col.startswith('L') and col.endswith('_uH')]
    
    z_data = df[z_columns].values  # Shape: (N_samples, F_num)
    l_data = df[l_columns].values  # Shape: (N_samples, N_coils)
    
    print(f"Data shape: Z={z_data.shape}, L={l_data.shape}")
    print(f"Z range: [{z_data.min():.2f}, {z_data.max():.2f}]")
    print(f"L range: [{l_data.min():.2f}, {l_data.max():.2f}] μH")
    
    # Split data
    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=test_size, random_state=random_state
    )
    
    z_train, z_val, l_train, l_val = train_test_split(
        z_temp, l_temp, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Normalize features
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)
    
    # Normalize targets (important for regression)
    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)
    
    print(f"Split sizes - Train: {len(z_train)}, Val: {len(z_val)}, Test: {len(z_test)}")
    
    return {
        'train': (z_train, l_train),
        'val': (z_val, l_val),
        'test': (z_test, l_test),
        'scalers': {'z_scaler': z_scaler, 'l_scaler': l_scaler}
    }


def main():
    """Main function to train Z-value network"""
    print("=== Z-Value Neural Network Training ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    try:
        data = load_and_prepare_data('z_dataset.csv')
    except FileNotFoundError:
        print("Error: z_dataset.csv not found. Please run task1_improved_dataset.py first.")
        return
    
    # Create datasets and data loaders
    train_dataset = ZImpedanceDataset(data['train'][0], data['train'][1])
    val_dataset = ZImpedanceDataset(data['val'][0], data['val'][1])
    test_dataset = ZImpedanceDataset(data['test'][0], data['test'][1])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = data['train'][0].shape[1]  # Number of impedance features
    model = ZValueNetwork(input_size=input_size)
    
    print(f"Model architecture:")
    print(f"Input size: {input_size}")
    print(f"Hidden layers: {[512, 256, 128, 64]}")
    print(f"Output size: 5")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = ZNetworkTrainer(model, device=device, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=20, save_path='models/z_network.pth')
    
    # Plot training curves
    trainer.plot_training_curves('results/z_network_training.png')
    
    # Evaluate on test set
    results = trainer.evaluate(test_loader, 'models/z_network.pth')
    
    print("\n✅ Z-Value network training completed successfully!")


if __name__ == "__main__":
    main()