import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

# Import our network modules
from task2_z_network import (
    ZValueNetwork, ZNetworkTrainer, ZImpedanceDataset,
    load_and_prepare_data
)
from task2_mixed_z_network import (
    MixedZValueNetwork, ComplexFeatureExtractor, MixedZNetworkTrainer,
    MixedZImpedanceDataset, load_and_prepare_mixed_data
)
from torch.utils.data import DataLoader


class NetworkOrchestrator:
    """Orchestrates training and comparison of both network types"""
    
    def __init__(self, results_dir='results', models_dir='models'):
        """
        Initialize the orchestrator
        
        Args:
            results_dir: Directory to save results and plots
            models_dir: Directory to save trained models
        """
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
        print(f"Neural Network Training Orchestrator")
        print(f"Device: {self.device}")
        print(f"Results directory: {results_dir}")
        print(f"Models directory: {models_dir}")
        print("=" * 60)
    
    def check_data_availability(self):
        """Check if required datasets are available"""
        z_dataset_path = 'z_dataset.csv'
        mixed_z_dataset_path = 'mixed_z_dataset.csv'
        
        if not os.path.exists(z_dataset_path):
            print(f"❌ Error: {z_dataset_path} not found!")
            print("Please run task1_improved_dataset.py first to generate datasets.")
            return False
            
        if not os.path.exists(mixed_z_dataset_path):
            print(f"❌ Error: {mixed_z_dataset_path} not found!")
            print("Please run task1_improved_dataset.py first to generate datasets.")
            return False
        
        # Check dataset dimensions
        z_df = pd.read_csv(z_dataset_path)
        mixed_df = pd.read_csv(mixed_z_dataset_path)
        
        print(f"✅ Z-dataset found: {z_df.shape}")
        print(f"✅ Mixed Z-dataset found: {mixed_df.shape}")
        
        return True
    
    def train_z_network(self, epochs=100, learning_rate=0.001, batch_size=32):
        """Train the Z-magnitude network"""
        print("\n🔵 Training Z-Value Network")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            data = load_and_prepare_data('z_dataset.csv')
            
            # Create datasets and loaders
            train_dataset = ZImpedanceDataset(data['train'][0], data['train'][1])
            val_dataset = ZImpedanceDataset(data['val'][0], data['val'][1])
            test_dataset = ZImpedanceDataset(data['test'][0], data['test'][1])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Create model
            input_size = data['train'][0].shape[1]
            model = ZValueNetwork(input_size=input_size)
            
            print(f"Model: Z-Value Network")
            print(f"Input size: {input_size}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            trainer = ZNetworkTrainer(model, device=self.device, learning_rate=learning_rate)
            model_path = os.path.join(self.models_dir, 'z_network.pth')
            trainer.train(train_loader, val_loader, epochs=epochs, save_path=model_path)
            
            # Plot training curves
            plot_path = os.path.join(self.results_dir, 'z_network_training.png')
            trainer.plot_training_curves(plot_path)
            
            # Evaluate
            results = trainer.evaluate(test_loader, model_path)
            
            # Store results
            training_time = time.time() - start_time
            self.training_results['z_network'] = {
                'training_time': training_time,
                'epochs': epochs,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': trainer.train_losses[-1],
                'final_val_loss': trainer.val_losses[-1],
                'input_size': input_size,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            self.evaluation_results['z_network'] = results
            
            print(f"✅ Z-Network training completed in {training_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Error training Z-Network: {e}")
            return False
    
    def train_mixed_z_network(self, epochs=100, learning_rate=0.0005, batch_size=16):
        """Train the Mixed Z-complex network"""
        print("\n🟡 Training Mixed Z-Value Network")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            data = load_and_prepare_mixed_data('mixed_z_dataset.csv')
            
            # Create datasets and loaders
            train_dataset = MixedZImpedanceDataset(data['train'][0], data['train'][1])
            val_dataset = MixedZImpedanceDataset(data['val'][0], data['val'][1])
            test_dataset = MixedZImpedanceDataset(data['test'][0], data['test'][1])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Create model
            input_size = data['train'][0].shape[1]
            model = MixedZValueNetwork(input_size=input_size)
            
            print(f"Model: Mixed Z-Value Network")
            print(f"Input size: {input_size}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            trainer = MixedZNetworkTrainer(model, device=self.device, learning_rate=learning_rate)
            model_path = os.path.join(self.models_dir, 'mixed_z_network.pth')
            trainer.train(train_loader, val_loader, epochs=epochs, save_path=model_path)
            
            # Plot training curves
            plot_path = os.path.join(self.results_dir, 'mixed_z_network_training.png')
            trainer.plot_training_curves(plot_path)
            
            # Evaluate
            results = trainer.evaluate(test_loader, model_path)
            
            # Store results
            training_time = time.time() - start_time
            self.training_results['mixed_z_network'] = {
                'training_time': training_time,
                'epochs': epochs,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': trainer.train_losses[-1],
                'final_val_loss': trainer.val_losses[-1],
                'input_size': input_size,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            self.evaluation_results['mixed_z_network'] = results
            
            print(f"✅ Mixed Z-Network training completed in {training_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Error training Mixed Z-Network: {e}")
            return False
    
    def train_complex_feature_network(self, epochs=100, learning_rate=0.0005, batch_size=16):
        """Train the Complex Feature Extractor network"""
        print("\n🟢 Training Complex Feature Extractor Network")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            data = load_and_prepare_mixed_data('mixed_z_dataset.csv')
            
            # Create datasets and loaders
            train_dataset = MixedZImpedanceDataset(data['train'][0], data['train'][1])
            val_dataset = MixedZImpedanceDataset(data['val'][0], data['val'][1])
            test_dataset = MixedZImpedanceDataset(data['test'][0], data['test'][1])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Create model
            input_size = data['train'][0].shape[1]
            f_num = input_size // 2
            model = ComplexFeatureExtractor(f_num=f_num)
            
            print(f"Model: Complex Feature Extractor")
            print(f"F_num: {f_num}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            trainer = MixedZNetworkTrainer(model, device=self.device, learning_rate=learning_rate)
            model_path = os.path.join(self.models_dir, 'complex_extractor_network.pth')
            trainer.train(train_loader, val_loader, epochs=epochs, save_path=model_path)
            
            # Plot training curves
            plot_path = os.path.join(self.results_dir, 'complex_extractor_training.png')
            trainer.plot_training_curves(plot_path)
            
            # Evaluate
            results = trainer.evaluate(test_loader, model_path)
            
            # Store results
            training_time = time.time() - start_time
            self.training_results['complex_extractor'] = {
                'training_time': training_time,
                'epochs': epochs,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': trainer.train_losses[-1],
                'final_val_loss': trainer.val_losses[-1],
                'f_num': f_num,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            self.evaluation_results['complex_extractor'] = results
            
            print(f"✅ Complex Feature Network training completed in {training_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Error training Complex Feature Network: {e}")
            return False
    
    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        print("\n📊 Generating Comparison Plots")
        print("-" * 40)
        
        # Model Performance Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.evaluation_results.keys())
        model_names = [name.replace('_', ' ').title() for name in models]
        
        # R² Score comparison
        r2_scores = [self.evaluation_results[model]['r2'] for model in models]
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(models)]
        
        bars1 = ax1.bar(model_names, r2_scores, color=colors, alpha=0.7)
        ax1.set_title('Model Performance - R² Score', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # MSE comparison
        mse_scores = [self.evaluation_results[model]['mse'] for model in models]
        bars2 = ax2.bar(model_names, mse_scores, color=colors, alpha=0.7)
        ax2.set_title('Model Performance - MSE', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, mse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{score:.2e}', ha='center', va='bottom', fontweight='bold', rotation=0)
        
        # Training time comparison
        training_times = [self.training_results[model]['training_time'] for model in models]
        bars3 = ax3.bar(model_names, training_times, color=colors, alpha=0.7)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars3, training_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Model complexity (parameters)
        param_counts = [self.training_results[model]['parameters'] for model in models]
        bars4 = ax4.bar(model_names, param_counts, color=colors, alpha=0.7)
        ax4.set_title('Model Complexity - Parameter Count', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Parameters')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, params in zip(bars4, param_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts)*0.01,
                    f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comparison plots saved to: {comparison_path}")
    
    def save_results_summary(self):
        """Save comprehensive results summary"""
        print("\n💾 Saving Results Summary")
        print("-" * 40)
        
        # Create comprehensive results dictionary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'training_results': self.training_results,
            'evaluation_results': {
                model: {k: float(v) if isinstance(v, np.ndarray) and v.ndim == 0 else v 
                       for k, v in results.items() if k not in ['predictions', 'targets']}
                for model, results in self.evaluation_results.items()
            }
        }
        
        # Save JSON summary
        json_path = os.path.join(self.results_dir, 'training_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create human-readable report
        report_path = os.path.join(self.results_dir, 'performance_report.md')
        with open(report_path, 'w') as f:
            f.write("# Neural Network Training Results\n\n")
            f.write(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device**: {self.device}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | R² Score | MSE | MAE | Training Time | Parameters |\n")
            f.write("|-------|----------|-----|-----|---------------|------------|\n")
            
            for model in self.evaluation_results.keys():
                eval_res = self.evaluation_results[model]
                train_res = self.training_results[model]
                model_name = model.replace('_', ' ').title()
                
                f.write(f"| {model_name} | {eval_res['r2']:.6f} | {eval_res['mse']:.2e} | "
                       f"{eval_res['mae']:.6f} | {train_res['training_time']:.1f}s | "
                       f"{train_res['parameters']:,} |\n")
            
            f.write("\n## Best Performing Model\n\n")
            
            # Find best model by R² score
            best_model = max(self.evaluation_results.keys(), 
                           key=lambda x: self.evaluation_results[x]['r2'])
            best_r2 = self.evaluation_results[best_model]['r2']
            
            f.write(f"**{best_model.replace('_', ' ').title()}** achieved the highest R² score of "
                   f"**{best_r2:.6f}**\n\n")
            
            f.write("## Training Details\n\n")
            for model, results in self.training_results.items():
                f.write(f"### {model.replace('_', ' ').title()}\n")
                f.write(f"- Training time: {results['training_time']:.1f} seconds\n")
                f.write(f"- Best validation loss: {results['best_val_loss']:.6f}\n")
                f.write(f"- Final training loss: {results['final_train_loss']:.6f}\n")
                f.write(f"- Final validation loss: {results['final_val_loss']:.6f}\n")
                f.write(f"- Model parameters: {results['parameters']:,}\n\n")
        
        print(f"✅ Results saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Report: {report_path}")
    
    def run_complete_training(self, epochs=100):
        """Run complete training pipeline for all networks"""
        print(f"\n🚀 Starting Complete Neural Network Training Pipeline")
        print(f"Training for {epochs} epochs per model")
        print("=" * 60)
        
        # Check data availability
        if not self.check_data_availability():
            return False
        
        total_start_time = time.time()
        successful_trainings = 0
        
        # Train all networks
        if self.train_z_network(epochs=epochs):
            successful_trainings += 1
        
        if self.train_mixed_z_network(epochs=epochs):
            successful_trainings += 1
            
        if self.train_complex_feature_network(epochs=epochs):
            successful_trainings += 1
        
        total_time = time.time() - total_start_time
        
        print(f"\n📋 Training Pipeline Summary")
        print("=" * 60)
        print(f"Total training time: {total_time:.1f} seconds")
        print(f"Successfully trained models: {successful_trainings}/3")
        
        if successful_trainings > 0:
            # Generate analysis
            self.generate_comparison_plots()
            self.save_results_summary()
            
            print(f"\n✅ Training pipeline completed successfully!")
            print(f"📁 Results saved to: {self.results_dir}/")
            print(f"🤖 Models saved to: {self.models_dir}/")
            
            return True
        else:
            print(f"\n❌ Training pipeline failed - no models trained successfully")
            return False


def main():
    """Main function to orchestrate neural network training"""
    print("Neural Network Training Orchestration System")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = NetworkOrchestrator()
    
    # Run complete training pipeline
    success = orchestrator.run_complete_training(epochs=100)
    
    if success:
        print(f"\n🎉 All neural networks trained and analyzed successfully!")
        print(f"\nNext steps:")
        print(f"1. Review results in results/ directory")
        print(f"2. Check model performance in performance_report.md")  
        print(f"3. Use trained models for inductance prediction")
    else:
        print(f"\n⚠️  Training completed with some issues. Check error messages above.")


if __name__ == "__main__":
    main()