#!/usr/bin/env python3
"""
Neural Network Architecture Visualization for Circuit Simulation Project

Creates simplified and clear diagrams for all three neural network architectures:
1. FCNN (Fully Connected Neural Network)  
2. CNN (Convolutional Neural Network)
3. PINN (Physics-Informed Neural Network)

All diagrams are saved to the results/ directory.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Define consistent colors
COLORS = {
    'input': '#3498db',      # Blue
    'hidden': '#9b59b6',     # Purple  
    'conv': '#e74c3c',       # Red
    'output': '#2ecc71',     # Green
    'physics': '#f39c12',    # Orange
    'connection': '#7f8c8d', # Gray
    'text': '#2c3e50'        # Dark blue
}

def create_fcnn_diagram():
    """Create simplified FCNN architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Network architecture - more detailed layer information for mixed_z_fcnn_training
    layer_info = [
        {'name': 'Input Layer\nComplex Z\n3000', 'type': 'input', 'width': 0.8},
        {'name': 'Hidden Layer 1\nFC + ReLU\n1024', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 2\nFC + ReLU\n512', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 3\nFC + ReLU\n256', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 4\nFC + ReLU\n128', 'type': 'hidden', 'width': 0.8},
        {'name': 'Output Layer\nLinear\nL₁-L₅ (5)', 'type': 'output', 'width': 0.8}
    ]
    
    x_positions = np.linspace(1, 12, len(layer_info))
    
    # Draw layers with consistent styling
    for i, (x, layer) in enumerate(zip(x_positions, layer_info)):
        # Determine color
        if layer['type'] == 'input':
            color = COLORS['input']
        elif layer['type'] == 'output':
            color = COLORS['output']
        else:
            color = COLORS['hidden']
        
        # Draw layer rectangle with consistent dimensions
        height = 3
        width = layer['width']
        rect = Rectangle((x-width/2, 2), width, height, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add layer name with consistent formatting
        ax.text(x, 3.5, layer['name'], ha='center', va='center', 
               fontsize=10, weight='bold', color='white')
        
        # Draw connections to next layer
        if i < len(layer_info) - 1:
            next_x = x_positions[i + 1]
            next_width = layer_info[i + 1]['width']
            # Draw multiple connection lines
            for j in range(5):
                y_start = 2.2 + j * 0.6
                y_end = 2.2 + j * 0.6
                ax.plot([x + width/2, next_x - next_width/2], [y_start, y_end], 
                       color=COLORS['connection'], alpha=0.5, linewidth=1)
    
    # Add activation functions and regularization annotations
    activation_positions = x_positions[1:-1]  # All hidden layers
    for x in activation_positions:
        ax.text(x, 1.2, 'ReLU\n+\nDropout(0.2)', ha='center', va='center', 
               fontsize=8, style='italic', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Add title and description
    ax.set_title('FCNN (Fully Connected Neural Network) Architecture', fontsize=16, weight='bold', pad=20)
    ax.text(6.5, 0.3, 'Input: Complex Impedance Z(ω) = Real + i·Imag → Fully Connected Layers → Output: Inductance Parameters L₁-L₅', 
           ha='center', va='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/fcnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ FCNN architecture diagram generated: results/fcnn_architecture.png")


def create_cnn_diagram():
    """Create simplified CNN architecture diagram - consistent with FCNN style"""
    
    fig, ax = plt.subplots(figsize=(14, 8))  # Same size as FCNN
    
    # Network layers - organized horizontally like FCNN
    layer_info = [
        {'name': 'Input Layer\nComplex Z\n[2,1500]', 'type': 'input', 'width': 0.8},
        {'name': 'Conv1D\n2→32\n+MaxPool', 'type': 'conv', 'width': 1.0},
        {'name': 'Conv1D\n32→64\n+MaxPool', 'type': 'conv', 'width': 1.0},
        {'name': 'Conv1D\n64→128\n+MaxPool', 'type': 'conv', 'width': 1.0},
        {'name': 'Flatten\n24K', 'type': 'flatten', 'width': 0.8},
        {'name': 'FC Layer\n256', 'type': 'hidden', 'width': 0.8},
        {'name': 'FC Layer\n128', 'type': 'hidden', 'width': 0.8},
        {'name': 'Output Layer\nL₁-L₅\n5', 'type': 'output', 'width': 0.8}
    ]
    
    x_positions = np.linspace(1, 12, len(layer_info))
    
    # Draw layers with consistent style
    for i, (x, layer) in enumerate(zip(x_positions, layer_info)):
        # Determine color based on layer type
        if layer['type'] == 'input':
            color = COLORS['input']
        elif layer['type'] == 'conv':
            color = COLORS['conv']
        elif layer['type'] == 'flatten':
            color = COLORS['physics']  # Orange for flatten
        elif layer['type'] == 'output':
            color = COLORS['output']
        else:  # hidden
            color = COLORS['hidden']
        
        # Draw layer rectangle with consistent dimensions
        height = 3  # Same as FCNN
        width = layer['width']
        rect = Rectangle((x-width/2, 2), width, height, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add layer name with consistent styling
        ax.text(x, 3.5, layer['name'], ha='center', va='center', 
               fontsize=10, weight='bold', color='white')
        
        # Draw connections to next layer (same style as FCNN)
        if i < len(layer_info) - 1:
            next_x = x_positions[i + 1]
            next_width = layer_info[i + 1]['width']
            # Draw multiple connection lines like FCNN
            for j in range(5):
                y_start = 2.2 + j * 0.6
                y_end = 2.2 + j * 0.6
                ax.plot([x + width/2, next_x - next_width/2], [y_start, y_end], 
                       color=COLORS['connection'], alpha=0.5, linewidth=1)
    
    # Add activation/operation annotations (consistent with FCNN style)
    conv_positions = x_positions[1:4]  # Conv layers
    for x in conv_positions:
        ax.text(x, 1.2, 'ReLU\n+\nMaxPool', ha='center', va='center', 
               fontsize=8, style='italic', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # FC layer activations
    fc_positions = x_positions[5:7]  # FC layers
    for x in fc_positions:
        ax.text(x, 1.2, 'ReLU\n+\nDropout', ha='center', va='center', 
               fontsize=8, style='italic', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # Add title and labels (consistent with FCNN)
    ax.set_title('CNN (Convolutional Neural Network) Architecture', fontsize=16, weight='bold', pad=20)
    ax.text(6.5, 0.3, 'Input: Complex Impedance (Real+Imag) → Conv Feature Extraction → FC Prediction → Output: Inductance Parameters', 
           ha='center', va='center', fontsize=12, style='italic')
    
    # Consistent dimensions and layout with FCNN
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/cnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ CNN architecture diagram generated: results/cnn_architecture.png")


def create_pinn_diagram():
    """Create three separate PINN architecture diagrams"""
    
    # Diagram 1: Neural Network Architecture
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Neural Network part - horizontal layout like other networks
    layer_info = [
        {'name': 'Input Layer\nComplex Z\n[3000]', 'type': 'input', 'width': 0.8},
        {'name': 'Hidden Layer 1\nFC + ReLU\n1024', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 2\nFC + ReLU\n512', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 3\nFC + ReLU\n256', 'type': 'hidden', 'width': 0.8},
        {'name': 'Hidden Layer 4\nFC + ReLU\n128', 'type': 'hidden', 'width': 0.8},
        {'name': 'Output Layer\nLinear\nL₁-L₅ (5)', 'type': 'output', 'width': 0.8}
    ]
    
    x_positions = np.linspace(1, 12, len(layer_info))
    
    # Draw layers with consistent styling
    for i, (x, layer) in enumerate(zip(x_positions, layer_info)):
        # Determine color
        if layer['type'] == 'input':
            color = COLORS['input']
        elif layer['type'] == 'output':
            color = COLORS['output']
        else:
            color = COLORS['hidden']
        
        # Draw layer rectangle with consistent dimensions
        height = 3
        width = layer['width']
        rect = Rectangle((x-width/2, 2), width, height, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add layer name with consistent formatting
        ax.text(x, 3.5, layer['name'], ha='center', va='center', 
               fontsize=10, weight='bold', color='white')
        
        # Draw connections to next layer
        if i < len(layer_info) - 1:
            next_x = x_positions[i + 1]
            next_width = layer_info[i + 1]['width']
            # Draw multiple connection lines
            for j in range(5):
                y_start = 2.2 + j * 0.6
                y_end = 2.2 + j * 0.6
                ax.plot([x + width/2, next_x - next_width/2], [y_start, y_end], 
                       color=COLORS['connection'], alpha=0.5, linewidth=1)
    
    # Add title
    ax.set_title('PINN Neural Network Architecture', fontsize=16, weight='bold', pad=20)
    ax.text(6.5, 0.5, 'Input: Complex Impedance → Fully Connected Network → Output: Inductance Parameters', 
           ha='center', va='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/pinn_network_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PINN network architecture diagram generated: results/pinn_network_architecture.png")
    
    # Diagram 2: Physics Constraints
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Physics constraints layout
    phys_y = 5.5
    
    # Left side - Input processing
    ax.add_patch(Rectangle((1, phys_y-1.5), 4, 3, 
                          facecolor=COLORS['input'], alpha=0.3, edgecolor=COLORS['input'], linewidth=3))
    ax.text(3, phys_y, 'Input Processing\n\nComplex Impedance\nZ(ω) = Real + i·Imag\n\nFrequency Domain Analysis\nPhase & Magnitude', 
           ha='center', va='center', fontsize=12, weight='bold', color=COLORS['input'])
    
    # Middle - Physics constraints
    ax.add_patch(Rectangle((5.5, phys_y-1.5), 5, 3, 
                          facecolor=COLORS['physics'], alpha=0.3, edgecolor=COLORS['physics'], linewidth=3))
    ax.text(8, phys_y, 'Physics Constraints\n\n• Resonance: ω₀ = 1/√(LC)\n• Quality Factor: Q = R/(ωL)\n• Impedance: Im(Z) ≈ 0\n• Mutual Coupling Effects', 
           ha='center', va='center', fontsize=12, weight='bold', color=COLORS['physics'])
    
    # Right side - Output validation
    ax.add_patch(Rectangle((11, phys_y-1.5), 4, 3, 
                          facecolor=COLORS['output'], alpha=0.3, edgecolor=COLORS['output'], linewidth=3))
    ax.text(13, phys_y, 'Output Validation\n\nInductance Parameters\nL₁, L₂, L₃, L₄, L₅\n\nPhysical Constraints\nPositive Values', 
           ha='center', va='center', fontsize=12, weight='bold', color=COLORS['output'])
    
    # Physics constraints (horizontal arrows)
    ax.annotate('', xy=(5.3, phys_y), xytext=(5.2, phys_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['physics']))
    ax.annotate('', xy=(10.8, phys_y), xytext=(10.7, phys_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['physics']))
    
    # Add title
    ax.set_title('PINN Physics Constraints & Validation', fontsize=16, weight='bold', pad=20)
    
    # Add key advantages
    advantages_text = (
        'Key Physics Integration Features:\n\n'
        '• Domain knowledge enforcement\n'
        '• Physical law compliance\n'
        '• Improved generalization\n'
        '• Reduced training data needs\n'
        '• Better extrapolation capability\n'
        '• Interpretable predictions'
    )
    ax.text(8, 2, advantages_text, ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9, edgecolor='green', linewidth=2))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0.5, 8)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/pinn_physics_constraints.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PINN physics constraints diagram generated: results/pinn_physics_constraints.png")
    
    # Diagram 3: Loss Function
    fig, ax = plt.subplots(figsize=(14, 6))
    
    loss_y = 3
    
    # MSE loss
    ax.add_patch(Rectangle((1, loss_y-1), 3.5, 2, 
                          facecolor=COLORS['conv'], alpha=0.3, edgecolor=COLORS['conv'], linewidth=3))
    ax.text(2.75, loss_y, 'Data Loss\n\n‖ŷ - y‖²\n(MSE)\n\nPrediction Accuracy', ha='center', va='center', 
           fontsize=12, weight='bold', color=COLORS['conv'])
    
    # Plus sign
    ax.text(5, loss_y, '+', ha='center', va='center', fontsize=24, weight='bold', color=COLORS['text'])
    
    # Physics loss
    ax.add_patch(Rectangle((5.75, loss_y-1), 3.5, 2, 
                          facecolor=COLORS['physics'], alpha=0.3, edgecolor=COLORS['physics'], linewidth=3))
    ax.text(7.5, loss_y, 'Physics Loss\n\nλ₁·L_res + λ₂·L_imp\n(Constraints)\n\nPhysical Compliance', ha='center', va='center', 
           fontsize=12, weight='bold', color=COLORS['physics'])
    
    # Equals sign
    ax.text(10, loss_y, '=', ha='center', va='center', fontsize=24, weight='bold', color=COLORS['text'])
    
    # Total loss
    ax.add_patch(Rectangle((10.75, loss_y-1), 3, 2, 
                          facecolor=COLORS['text'], alpha=0.3, edgecolor=COLORS['text'], linewidth=3))
    ax.text(12.25, loss_y, 'Total Loss\n\nL_total\n(Combined)\n\nOptimization Target', ha='center', va='center', 
           fontsize=12, weight='bold', color=COLORS['text'])
    
    # Add title
    ax.set_title('PINN Loss Function Composition', fontsize=16, weight='bold', pad=20)
    
    # Add formula
    formula_text = 'L_total = L_data + λ₁ × L_resonance + λ₂ × L_impedance'
    ax.text(7.5, 1, formula_text, ha='center', va='center', fontsize=14, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/pinn_loss_function.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PINN loss function diagram generated: results/pinn_loss_function.png")


def create_comparison_diagram():
    """Create a comparison diagram of all three architectures"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    # FCNN (top)
    ax1.text(0.5, 0.5, 'FCNN', ha='left', va='center', fontsize=14, weight='bold', color=COLORS['input'])
    ax1.text(2, 0.5, 'Input: Impedance Magnitude |Z(ω)|', ha='left', va='center', fontsize=12)
    ax1.text(6, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax1.text(7, 0.5, 'Fully Connected [512→256→128→64]', ha='left', va='center', fontsize=12)
    ax1.text(12, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax1.text(13, 0.5, 'Output: L₁-L₅', ha='left', va='center', fontsize=12)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # CNN (middle)
    ax2.text(0.5, 0.5, 'CNN', ha='left', va='center', fontsize=14, weight='bold', color=COLORS['conv'])
    ax2.text(2, 0.5, 'Input: Complex Impedance (Real+Imag)', ha='left', va='center', fontsize=12)
    ax2.text(6, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax2.text(7, 0.5, 'Conv+Pooling [32→64→128]', ha='left', va='center', fontsize=12)
    ax2.text(10.5, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax2.text(11.5, 0.5, 'FC [256→128] → L₁-L₅', ha='left', va='center', fontsize=12)
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # PINN (bottom)
    ax3.text(0.5, 0.5, 'PINN', ha='left', va='center', fontsize=14, weight='bold', color=COLORS['physics'])
    ax3.text(2, 0.5, 'Input: Complex Impedance', ha='left', va='center', fontsize=12)
    ax3.text(4.5, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax3.text(5, 0.5, 'FCNN Architecture', ha='left', va='center', fontsize=12)
    ax3.text(7.5, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax3.text(8, 0.5, 'Physics Constraint Loss', ha='left', va='center', fontsize=12, color=COLORS['physics'])
    ax3.text(11, 0.5, '→', ha='center', va='center', fontsize=16, weight='bold')
    ax3.text(11.5, 0.5, 'L₁-L₅', ha='left', va='center', fontsize=12)
    ax3.set_xlim(0, 16)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.suptitle('Neural Network Architecture Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('results/network_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Network comparison diagram generated: results/network_comparison.png")


def main():
    """Generate all neural network architecture diagrams"""
    
    print("Starting neural network architecture diagram generation...")
    
    # Create individual architecture diagrams
    create_fcnn_diagram()
    create_cnn_diagram()  
    create_pinn_diagram()
    
    # Create comparison diagram
    create_comparison_diagram()
    
    print("\n🎉 All architecture diagrams generated successfully!")
    print("📁 File location: results/")
    print("   - fcnn_architecture.png (FCNN Architecture Diagram)")
    print("   - cnn_architecture.png (CNN Architecture Diagram)")
    print("   - pinn_network_architecture.png (PINN Network Architecture)")
    print("   - pinn_physics_constraints.png (PINN Physics Constraints)")
    print("   - pinn_loss_function.png (PINN Loss Function)")
    print("   - network_comparison.png (Architecture Comparison)")


if __name__ == "__main__":
    main()