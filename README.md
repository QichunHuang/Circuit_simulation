# Circuit Simulation

A comprehensive machine learning framework for analyzing and predicting inductance parameters in coupled resonator circuits using neural networks.

## Overview

This project provides tools for simulating coupled resonator circuits and using deep learning to predict inductance values from impedance spectral data. It includes dataset generation, multiple neural network architectures, and comprehensive analysis tools.

## Features

- **Circuit Simulation**: Accurate modeling of coupled resonator circuits with configurable parameters
- **Dataset Generation**: Automated generation of impedance spectra with variable inductance parameters  
- **Neural Network Models**: Multiple architectures for impedance-to-inductance prediction
- **Comprehensive Analysis**: Performance comparison, activation function studies, and visualization

## Project Structure

```
├── task1_improved_dataset.py          # Enhanced dataset generation with systematic sampling
├── task2_z_network.py                 # Basic Z-magnitude neural network
├── task2_mixed_z_network.py          # Mixed Z-magnitude and phase network
├── task2_complex_feature_network.py   # Network with complex impedance features
├── task2_cnn_network.py              # Convolutional neural network approach
├── task2_pinn_network.py             # Physics-informed neural network
├── task3_train_networks.py           # Batch training script for all models
├── coupled_resonator_tools.py        # Core circuit simulation functions
├── neural_network_diagrams.py        # Network architecture visualization
├── experiment*.py                     # Comparative analysis experiments
├── stage4_comprehensive_analysis.py  # Final comprehensive evaluation
└── *.csv                            # Generated datasets
```

## Dependencies

The project requires Python ≥3.12 and the following packages:

```toml
- torch >= 2.8.0          # PyTorch for neural networks
- numpy >= 2.3.2          # Numerical computations
- pandas >= 2.3.1         # Data manipulation
- scikit-learn >= 1.7.1   # ML utilities and metrics
- scipy >= 1.16.1         # Scientific computing
- matplotlib >= 3.10.5    # Plotting
- seaborn >= 0.13.2       # Statistical visualization
- plotly >= 6.3.0         # Interactive plots
- bokeh >= 3.7.3          # Web-based visualization
- tqdm >= 4.67.1          # Progress bars
- tensorboard >= 2.20.0   # Training monitoring
- wandb >= 0.21.1         # Experiment tracking
```

## Installation

1. Clone the repository
2. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Usage

### 1. Generate Dataset

Create training datasets with variable inductance parameters:

```python
from task1_improved_dataset import generate_dataset_with_variations

# Generate dataset with systematic sampling
generate_dataset_with_variations(
    L_num=500,              # Number of samples
    variation_index=3,      # Which inductance to vary (0-4)
    variation_percent=0.1   # ±10% variation
)
```

### 2. Train Models

Train individual models:

```python
from task2_z_network import train_z_network

# Train basic Z-magnitude network
model, metrics = train_z_network(
    dataset_index=0,
    epochs=200,
    batch_size=32
)
```

Or train all models at once:

```bash
python task3_train_networks.py
```

### 3. Run Experiments

Compare different approaches:

```python
# Compare single vs multiple inductance variations
python experiment1_single_L.py

# Compare network architectures  
python experiment2_network_compare.py

# Compare activation functions
python experiment3_activation_compare.py
```

## Network Architectures

### 1. Basic Z-Network (`task2_z_network.py`)
- Input: Impedance magnitude spectrum (1000 frequencies)
- Architecture: Fully connected layers [512, 256, 128, 64]
- Output: 5 inductance values

### 2. Mixed Z-Network (`task2_mixed_z_network.py`)
- Input: Both magnitude and phase spectra (2000 features)
- Enhanced feature representation for better accuracy

### 3. Complex Feature Network (`task2_complex_feature_network.py`)
- Input: Real and imaginary impedance components
- Captures full complex impedance information

### 4. CNN Network (`task2_cnn_network.py`)
- 1D convolutional layers for frequency domain patterns
- Automatic feature extraction from spectra

### 5. Physics-Informed Network (`task2_pinn_network.py`)
- Incorporates circuit physics in loss function
- Enforces physical constraints during training

## Key Functions

### Circuit Simulation (`coupled_resonator_tools.py`)

```python
# Calculate impedance spectrum
z_spectrum = calculate_impedance_spectrum(params)

# Build inductance matrix
M = build_inductance_matrix(params)

# Calculate resonant modes
frequencies, modes = calculate_modes(params)
```

### Dataset Generation (`task1_improved_dataset.py`)

```python
# Generate coupling matrix
K = generate_coupling_matrix_matlab_style(N=5, kk=0.01)

# Create inductance variations
L_variations = generate_L_variations(
    L_base=[1e-6] * 5,
    L_num=500,
    variation_index=3,
    variation_percent=0.1
)
```

## Results

The project generates comprehensive performance metrics including:

- **Regression Metrics**: MSE, MAE, R² score
- **Training Curves**: Loss and accuracy over epochs  
- **Model Comparison**: Performance across different architectures
- **Activation Analysis**: Impact of different activation functions

## Visualization

The framework provides extensive visualization capabilities:

- Network architecture diagrams
- Training progress plots
- Performance comparison charts
- Impedance spectrum visualizations
- Error distribution analysis

## Applications

This framework is suitable for:

- **Circuit Design**: Predicting inductance parameters from measurements
- **Quality Control**: Automated verification of circuit parameters
- **Research**: Studying coupled resonator behavior
- **Education**: Understanding circuit-ML integration

## Contributing

Contributions are welcome! Please focus on:

- New network architectures
- Improved physics constraints
- Enhanced visualization tools
- Performance optimizations

## License

This project is available for research and educational purposes.