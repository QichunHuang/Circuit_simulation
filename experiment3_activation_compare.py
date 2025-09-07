#!/usr/bin/env python3
"""
Experiment 3: Activation Function Comparison

Objective: Compare different activation functions using L3 dataset
- ReLU: Current baseline activation function (from experiment2)
- LeakyReLU: Mitigate dying ReLU problem with small negative slope
- SiLU: Smooth activation with self-gating mechanism

Expected Results:
- Performance comparison across activation functions
- Training convergence analysis
- Activation-specific advantages and limitations
"""

import copy
import json
import os
import time
from pathlib import Path

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


# ============================================================================
# Dataset Classes
# ============================================================================

class ComplexImpedanceDataset(Dataset):
    """Complex impedance dataset for activation function comparison"""
    
    def __init__(self, z_data, l_data, transform=None):
        """Initialize dataset
        
        Args:
            z_data: Complex impedance data (N_samples, 3000)
            l_data: Inductance targets (N_samples, 5) 
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


# ============================================================================
# Network Architectures with Different Activation Functions
# ============================================================================

class FCNNBase(nn.Module):
    """Base FCNN architecture with configurable activation function"""
    
    def __init__(self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128], 
                 activation_type='ReLU'):
        """Initialize FCNN with specified activation
        
        Args:
            input_size: Input feature dimensions
            output_size: Output dimensions (inductance parameters)
            hidden_sizes: Hidden layer neuron counts
            activation_type: Type of activation function ('ReLU', 'LeakyReLU', 'SiLU')
        """
        super(FCNNBase, self).__init__()
        
        self.activation_type = activation_type
        
        # Select activation function
        if activation_type == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_type == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation,
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ReLUFCNN(FCNNBase):
    """FCNN with ReLU activation (baseline)"""
    
    def __init__(self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]):
        super(ReLUFCNN, self).__init__(input_size, output_size, hidden_sizes, 'ReLU')


class LeakyReLUFCNN(FCNNBase):
    """FCNN with LeakyReLU activation"""
    
    def __init__(self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]):
        super(LeakyReLUFCNN, self).__init__(input_size, output_size, hidden_sizes, 'LeakyReLU')


class SiLUFCNN(FCNNBase):
    """FCNN with SiLU (Swish) activation"""
    
    def __init__(self, input_size=3000, output_size=5, hidden_sizes=[1024, 512, 256, 128]):
        super(SiLUFCNN, self).__init__(input_size, output_size, hidden_sizes, 'SiLU')


# ============================================================================
# Training Classes
# ============================================================================

class ActivationTrainer:
    """Generic network trainer for activation function comparison"""
    
    def __init__(self, model, activation_name, device="cpu", learning_rate=0.001):
        """Initialize trainer
        
        Args:
            model: Network model instance
            activation_name: Name of the activation function
            device: Training device
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.activation_name = activation_name
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
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
    
    def train(self, train_loader, val_loader, epochs=200):
        """Complete training loop"""
        print(f"\n=== Training {self.activation_name} Network ===")
        print(f"Device: {self.device}")
        print(f"Max epochs: {epochs}")
        
        # Early stopping parameters
        patience = 20
        patience_counter = 0
        start_time = time.time()
        
        for epoch in tqdm(range(epochs), desc=f"Training {self.activation_name}"):
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
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"{self.activation_name} Epoch {epoch + 1:3d}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"{self.activation_name} early stopped at epoch {epoch + 1}")
                break
        
        training_time = time.time() - start_time
        
        print(f"{self.activation_name} training completed:")
        print(f"  - Best validation loss: {self.best_val_loss:.6f}")
        print(f"  - Epochs trained: {len(self.train_losses)}")
        print(f"  - Training time: {training_time:.2f} seconds")
        
        return {
            'activation_name': self.activation_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses),
            'training_time': training_time
        }
    
    def evaluate(self, test_loader):
        """Model evaluation"""
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
        
        print(f"\n=== {self.activation_name} Evaluation Results ===")
        print(f"Test MSE:  {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        print(f"Test MAE:  {mae:.6f}")
        print(f"Test R²:   {r2:.6f}")
        
        # Per-coil performance
        coil_names = ["L1", "L2", "L3", "L4", "L5"]
        coil_metrics = {}
        print("\\nPer-coil performance:")
        for i, name in enumerate(coil_names):
            coil_mse = mean_squared_error(targets[:, i], predictions[:, i])
            coil_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            coil_r2 = r2_score(targets[:, i], predictions[:, i])
            coil_metrics[name] = {'mse': coil_mse, 'mae': coil_mae, 'r2': coil_r2}
            print(f"  {name} - MSE: {coil_mse:.6f}, MAE: {coil_mae:.6f}, R²: {coil_r2:.6f}")
        
        return {
            'activation_name': self.activation_name,
            'predictions': predictions,
            'targets': targets,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coil_metrics': coil_metrics
        }


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_l3_dataset(dataset_path="mixed_z_dataset_2.csv"):
    """Load L3 dataset for activation function comparison
    
    Args:
        dataset_path: Path to L3 dataset
        
    Returns:
        tuple: Training, validation, test data and scalers
    """
    print(f"Loading L3 dataset for activation function comparison: {dataset_path}")
    
    # Load dataset using Polars
    df = pl.read_csv(dataset_path)
    
    # Extract features and targets
    z_columns = [col for col in df.columns if col.startswith('z_') and ('real' in col or 'imag' in col)]
    l_columns = [col for col in df.columns if col.startswith('L') and len(col) == 2]
    
    z_data = df.select(z_columns).to_numpy()
    l_data = df.select(l_columns).to_numpy()
    
    print(f"Dataset loaded successfully:")
    print(f"  - Complex impedance features: {z_data.shape}")
    print(f"  - Inductance targets: {l_data.shape}")
    print(f"  - Primary variation: L3")
    
    # Data splitting
    z_temp, z_test, l_temp, l_test = train_test_split(
        z_data, l_data, test_size=0.2, random_state=42
    )
    z_train, z_val, l_train, l_val = train_test_split(
        z_temp, l_temp, test_size=0.25, random_state=42
    )
    
    # Data normalization
    z_scaler = StandardScaler()
    z_train = z_scaler.fit_transform(z_train)
    z_val = z_scaler.transform(z_val)
    z_test = z_scaler.transform(z_test)
    
    l_scaler = StandardScaler()
    l_train = l_scaler.fit_transform(l_train)
    l_val = l_scaler.transform(l_val)
    l_test = l_scaler.transform(l_test)
    
    print(f"Data split completed:")
    print(f"  - Training samples: {len(z_train)}")
    print(f"  - Validation samples: {len(z_val)}")
    print(f"  - Test samples: {len(z_test)}")
    
    return {
        "train": (z_train, l_train),
        "val": (z_val, l_val),
        "test": (z_test, l_test),
        "scalers": {"z_scaler": z_scaler, "l_scaler": l_scaler}
    }


def create_data_loaders(train_data, val_data, test_data, batch_size=64):
    """Create data loaders for training
    
    Args:
        train_data: (z_train, l_train)
        val_data: (z_val, l_val)
        test_data: (z_test, l_test)
        batch_size: Batch size
        
    Returns:
        dict: Data loaders for train/val/test
    """
    train_dataset = ComplexImpedanceDataset(train_data[0], train_data[1])
    val_dataset = ComplexImpedanceDataset(val_data[0], val_data[1])
    test_dataset = ComplexImpedanceDataset(test_data[0], test_data[1])
    
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    return loaders


# ============================================================================
# Visualization and Analysis
# ============================================================================

def plot_training_comparison(all_results, save_path="results/experiment3_training_comparison.png"):
    """Plot training curves comparison for all activation functions"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'ReLU': 'blue', 'LeakyReLU': 'green', 'SiLU': 'red'}
    
    # Training loss comparison
    for result in all_results:
        name = result['activation_name']
        epochs = range(1, len(result['train_losses']) + 1)
        ax1.plot(epochs, result['train_losses'], 
                label=f'{name} Train', color=colors[name], alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Validation loss comparison
    for result in all_results:
        name = result['activation_name']
        epochs = range(1, len(result['val_losses']) + 1)
        ax2.plot(epochs, result['val_losses'], 
                label=f'{name} Val', color=colors[name], alpha=0.8, 
                linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Training convergence analysis
    for result in all_results:
        name = result['activation_name']
        epochs = range(1, len(result['train_losses']) + 1)
        ax3.plot(epochs, result['train_losses'], 
                label=f'{name}', color=colors[name], alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (Linear Scale)')
    ax3.set_title('Training Convergence Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final performance metrics bar chart
    activations = [result['activation_name'] for result in all_results]
    best_val_losses = [result['best_val_loss'] for result in all_results]
    
    bars = ax4.bar(activations, best_val_losses, color=[colors[name] for name in activations], alpha=0.7)
    ax4.set_ylabel('Best Validation Loss')
    ax4.set_title('Final Performance Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_val_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training comparison plots saved to: {save_path}")


def plot_performance_comparison(all_eval_results, save_path="results/experiment3_performance_comparison.png"):
    """Plot detailed performance comparison"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'ReLU': 'blue', 'LeakyReLU': 'green', 'SiLU': 'red'}
    
    activations = [result['activation_name'] for result in all_eval_results]
    
    # Performance metrics comparison
    metrics = ['mse', 'rmse', 'mae', 'r2']
    metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
    
    x = np.arange(len(activations))
    width = 0.2
    
    # Plot each metric
    for i, (metric, name) in enumerate(zip(metrics[:3], metric_names[:3])):
        values = [result[metric] for result in all_eval_results]
        ax1.bar(x + i*width, values, width, label=name, alpha=0.7)
    
    ax1.set_xlabel('Activation Function')
    ax1.set_ylabel('Error Metrics')
    ax1.set_title('Error Metrics Comparison (Lower is Better)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(activations)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # R² comparison (separate because higher is better)
    r2_values = [result['r2'] for result in all_eval_results]
    bars = ax2.bar(activations, r2_values, color=[colors[name] for name in activations], alpha=0.7)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison (Higher is Better)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{r2:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # Per-coil performance heatmap
    coil_names = ["L1", "L2", "L3", "L4", "L5"]
    r2_matrix = []
    
    for result in all_eval_results:
        coil_r2 = [result['coil_metrics'][coil]['r2'] for coil in coil_names]
        r2_matrix.append(coil_r2)
    
    r2_matrix = np.array(r2_matrix)
    
    im = ax3.imshow(r2_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(coil_names)))
    ax3.set_xticklabels(coil_names)
    ax3.set_yticks(range(len(activations)))
    ax3.set_yticklabels(activations)
    ax3.set_xlabel('Inductance Parameters')
    ax3.set_ylabel('Activation Function')
    ax3.set_title('Per-Coil R² Performance Heatmap')
    
    # Add text annotations
    for i in range(len(activations)):
        for j in range(len(coil_names)):
            ax3.text(j, i, f'{r2_matrix[i, j]:.3f}', ha='center', va='center',
                    color='white' if r2_matrix[i, j] < 0.5 else 'black', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('R² Score')
    
    # Activation function properties comparison
    properties = {
        'ReLU': {'Gradient_Flow': 1.0, 'Smoothness': 0.0, 'Non_Linearity': 0.5},
        'LeakyReLU': {'Gradient_Flow': 0.9, 'Smoothness': 0.3, 'Non_Linearity': 0.6},
        'SiLU': {'Gradient_Flow': 0.8, 'Smoothness': 1.0, 'Non_Linearity': 0.9}
    }
    
    property_names = list(properties['ReLU'].keys())
    x_pos = np.arange(len(property_names))
    width = 0.25
    
    for i, activation in enumerate(activations):
        values = [properties[activation][prop] for prop in property_names]
        ax4.bar(x_pos + i*width, values, width, label=activation, 
               color=colors[activation], alpha=0.7)
    
    ax4.set_xlabel('Activation Properties')
    ax4.set_ylabel('Property Score (0-1)')
    ax4.set_title('Activation Function Properties')
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels([prop.replace('_', ' ') for prop in property_names])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance comparison plots saved to: {save_path}")


def save_experiment3_results(all_training_results, all_eval_results, save_path="results/experiment3_results.json"):
    """Save experiment 3 results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create comprehensive summary
    summary = {
        'experiment_name': 'Experiment 3: Activation Function Comparison',
        'activations_compared': len(all_eval_results),
        'dataset_used': 'L3 variation dataset (mixed_z_dataset_2.csv)',
        'training_summary': [],
        'evaluation_summary': [],
        'performance_ranking': []
    }
    
    # Combine training and evaluation results
    for train_result, eval_result in zip(all_training_results, all_eval_results):
        # Training summary
        train_summary = {
            'activation_name': train_result['activation_name'],
            'epochs_trained': train_result['epochs_trained'],
            'best_val_loss': train_result['best_val_loss'],
            'training_time': train_result['training_time'],
            'converged': train_result['epochs_trained'] < 200  # Assume 200 was max
        }
        summary['training_summary'].append(train_summary)
        
        # Evaluation summary
        eval_summary = {
            'activation_name': eval_result['activation_name'],
            'test_mse': eval_result['mse'],
            'test_rmse': eval_result['rmse'],
            'test_mae': eval_result['mae'],
            'test_r2': eval_result['r2'],
            'coil_metrics': eval_result['coil_metrics']
        }
        summary['evaluation_summary'].append(eval_summary)
    
    # Performance ranking
    eval_sorted = sorted(all_eval_results, key=lambda x: x['r2'], reverse=True)
    for i, result in enumerate(eval_sorted, 1):
        summary['performance_ranking'].append({
            'rank': i,
            'activation_name': result['activation_name'],
            'r2_score': result['r2'],
            'rmse': result['rmse']
        })
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment 3 results saved to: {save_path}")


# ============================================================================
# Main Experiment Function
# ============================================================================

def main():
    """Main function for Experiment 3: Activation Function Comparison"""
    print("="*70)
    print("Experiment 3: Activation Function Comparison")
    print("="*70)
    print("Objective: Compare ReLU, LeakyReLU, and SiLU activation functions using L3 dataset")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load L3 dataset
    try:
        data = load_l3_dataset("mixed_z_dataset_2.csv")
    except FileNotFoundError:
        print("Error: mixed_z_dataset_2.csv not found.")
        print("Please run: uv run task1_improved_dataset.py first to generate datasets.")
        return
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data["train"], data["val"], data["test"], batch_size=64
    )
    
    # Initialize networks with different activation functions
    networks = {
        'ReLU': ReLUFCNN(input_size=3000, output_size=5),
        'LeakyReLU': LeakyReLUFCNN(input_size=3000, output_size=5),
        'SiLU': SiLUFCNN(input_size=3000, output_size=5)
    }
    
    # Print network information
    print("\\n" + "="*70)
    print("Activation Function Details")
    print("="*70)
    
    for name, model in networks.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name} Network:")
        print(f"  - Parameters: {param_count:,}")
        print(f"  - Architecture: 3000 → 1024 → 512 → 256 → 128 → 5")
        print(f"  - Activation: {name}")
        if name == 'ReLU':
            print("  - Properties: Fast computation, can cause dying neurons")
        elif name == 'LeakyReLU':
            print("  - Properties: Prevents dying neurons, slight negative slope (0.01)")
        elif name == 'SiLU':
            print("  - Properties: Smooth, self-gating, better gradient flow")
    
    # Train all networks
    all_training_results = []
    all_eval_results = []
    
    for name, model in networks.items():
        print(f"\\n{'='*50}")
        print(f"Training {name} Network")
        print(f"{'='*50}")
        
        try:
            # Create trainer
            trainer = ActivationTrainer(
                model=model,
                activation_name=name,
                device=device,
                learning_rate=0.001
            )
            
            # Train the network
            training_result = trainer.train(
                data_loaders['train'], 
                data_loaders['val'], 
                epochs=200
            )
            
            # Evaluate the network
            eval_result = trainer.evaluate(data_loaders['test'])
            
            # Save model
            model_path = f"models/experiment3_{name.lower()}_network.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            save_dict = {
                'model_state_dict': trainer.best_model_state,
                'activation_name': name,
                'training_result': training_result,
                'eval_result': eval_result,
                'model_config': {
                    'activation_type': name,
                    'input_size': 3000,
                    'output_size': 5,
                    'hidden_sizes': [1024, 512, 256, 128]
                }
            }
            
            torch.save(save_dict, model_path)
            print(f"{name} model saved to: {model_path}")
            
            # Store results
            all_training_results.append(training_result)
            all_eval_results.append(eval_result)
            
        except Exception as e:
            print(f"❌ {name} training failed: {str(e)}")
            continue
    
    # Generate comparison analysis
    if all_training_results:
        print(f"\\n{'='*70}")
        print("Experiment 3 Results Analysis")
        print(f"{'='*70}")
        
        # Create visualizations
        plot_training_comparison(all_training_results)
        plot_performance_comparison(all_eval_results)
        save_experiment3_results(all_training_results, all_eval_results)
        
        # Print performance summary
        print("\\n=== Activation Function Comparison Summary ===")
        print(f"{'Activation':<12} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Epochs':<8} {'Time(s)':<10}")
        print("-" * 70)
        
        # Sort by R² score for ranking
        combined_results = []
        for train_result, eval_result in zip(all_training_results, all_eval_results):
            combined_results.append({
                'activation': train_result['activation_name'],
                'r2': eval_result['r2'],
                'rmse': eval_result['rmse'],
                'mae': eval_result['mae'],
                'epochs': train_result['epochs_trained'],
                'time': train_result['training_time']
            })
        
        # Sort by R² (descending)
        combined_results.sort(key=lambda x: x['r2'], reverse=True)
        
        for i, result in enumerate(combined_results, 1):
            print(f"{result['activation']:<12} {result['r2']:<10.6f} {result['rmse']:<10.6f} "
                  f"{result['mae']:<10.6f} {result['epochs']:<8} {result['time']:<10.2f}")
        
        # Determine winner
        best_activation = combined_results[0]['activation']
        best_r2 = combined_results[0]['r2']
        
        print(f"\\n🏆 Best Performing Activation: {best_activation}")
        print(f"   R² Score: {best_r2:.6f}")
        
        # Activation function insights
        print("\\n=== Activation Function Insights ===")
        if best_activation == 'ReLU':
            print("   ✅ ReLU remains effective for this circuit simulation task")
            print("   ✅ Fast computation with good performance")
        elif best_activation == 'LeakyReLU':
            print("   ✅ LeakyReLU prevents dying neuron problems")
            print("   ✅ Small negative slope helps gradient flow")
        elif best_activation == 'SiLU':
            print("   ✅ SiLU provides smooth gradients and self-gating")
            print("   ✅ Better handling of complex impedance relationships")
        
        print("\\n✅ Experiment 3: Activation Function Comparison completed!")
        print(f"   Successfully compared {len(all_training_results)}/3 activation functions")
        print("   Result files:")
        print("   - Training comparison: results/experiment3_training_comparison.png")
        print("   - Performance comparison: results/experiment3_performance_comparison.png")
        print("   - Experiment results: results/experiment3_results.json")
        print("   - Trained models: models/experiment3_*_network.pth")
    else:
        print("\\n❌ Experiment 3 failed: No networks trained successfully")


if __name__ == "__main__":
    main()