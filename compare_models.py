"""
Model comparison and evaluation script
Compares multiple models and generates comprehensive reports
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

from train import train_model


class ModelComparator:
    """Compare multiple models on the same dataset"""

    def __init__(self, results_dir='./comparison_results'):
        """
        Args:
            results_dir: directory to save comparison results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.model_results = {}

    def train_all_models(self, model_names, data_dir='./tactile_data',
                        batch_size=32, num_epochs=100, learning_rate=0.001):
        """
        Train all specified models

        Args:
            model_names: list of model names to train
            data_dir: directory containing training data
            batch_size: batch size for training
            num_epochs: number of training epochs
            learning_rate: learning rate
        """
        print("\n" + "=" * 60)
        print(f"Training {len(model_names)} models for comparison")
        print("=" * 60)

        for i, model_name in enumerate(model_names):
            print(f"\n[{i+1}/{len(model_names)}] Training {model_name}...")

            try:
                results = train_model(
                    model_name=model_name,
                    data_dir=data_dir,
                    save_dir=os.path.join(self.results_dir, model_name),
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate
                )

                self.model_results[model_name] = results

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"Completed training {len(self.model_results)} models")
        print("=" * 60)

    def load_results_from_files(self, model_dirs):
        """
        Load results from previously trained models

        Args:
            model_dirs: dict mapping model names to their result directories
        """
        for model_name, model_dir in model_dirs.items():
            results_file = os.path.join(model_dir, 'results.json')

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.model_results[model_name] = json.load(f)
                print(f"Loaded results for {model_name}")
            else:
                print(f"Results file not found for {model_name}: {results_file}")

    def generate_comparison_table(self):
        """Generate comparison table of all models"""
        if not self.model_results:
            print("No results to compare")
            return None

        comparison_data = []

        for model_name, results in self.model_results.items():
            test_results = results['test_results']

            row = {
                'Model': model_name.upper(),
                'Parameters': results['num_parameters'],
                'Accuracy': test_results['accuracy'],
                'Precision': test_results['precision'],
                'Recall': test_results['recall'],
                'F1-Score': test_results['f1_score'],
                'Best Val Acc': max(results['training_history']['val_acc']),
                'Total Epochs': len(results['training_history']['train_loss']),
                'Total Time (min)': sum(results['training_history']['epoch_time']) / 60
            }

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by test accuracy
        df = df.sort_values('Accuracy', ascending=False)

        return df

    def plot_comparison(self):
        """Generate comparison plots"""
        if not self.model_results:
            print("No results to compare")
            return

        # Prepare data
        model_names = list(self.model_results.keys())
        accuracies = [r['test_results']['accuracy'] for r in self.model_results.values()]
        precisions = [r['test_results']['precision'] for r in self.model_results.values()]
        recalls = [r['test_results']['recall'] for r in self.model_results.values()]
        f1_scores = [r['test_results']['f1_score'] for r in self.model_results.values()]
        num_params = [r['num_parameters'] for r in self.model_results.values()]

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Metrics comparison
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(model_names))
        width = 0.2

        ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax1.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in model_names], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # 2. Accuracy vs Parameters
        ax2 = plt.subplot(2, 3, 2)
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

        for i, (name, acc, params) in enumerate(zip(model_names, accuracies, num_params)):
            ax2.scatter(params, acc, s=200, c=[colors[i]], alpha=0.7, edgecolors='black')
            ax2.annotate(name.upper(), (params, acc), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Accuracy vs Model Complexity')
        ax2.grid(True, alpha=0.3)

        # 3. Training curves comparison (accuracy)
        ax3 = plt.subplot(2, 3, 3)
        for model_name in model_names:
            val_acc = self.model_results[model_name]['training_history']['val_acc']
            ax3.plot(val_acc, label=model_name.upper(), linewidth=2)

        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Accuracy')
        ax3.set_title('Validation Accuracy During Training')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Training time comparison
        ax4 = plt.subplot(2, 3, 4)
        training_times = [sum(r['training_history']['epoch_time'])/60
                         for r in self.model_results.values()]

        bars = ax4.barh(model_names, training_times, color=colors, alpha=0.7)
        ax4.set_xlabel('Training Time (minutes)')
        ax4.set_ylabel('Model')
        ax4.set_title('Training Time Comparison')
        ax4.set_yticklabels([m.upper() for m in model_names])
        ax4.grid(True, alpha=0.3, axis='x')

        # Add values on bars
        for i, (bar, time) in enumerate(zip(bars, training_times)):
            ax4.text(time, bar.get_y() + bar.get_height()/2,
                    f'{time:.1f}m', ha='left', va='center', fontsize=9)

        # 5. Per-class F1 scores comparison (for first model as reference)
        ax5 = plt.subplot(2, 3, 5)
        first_model = list(self.model_results.keys())[0]
        label_names = self.model_results[first_model].get('label_names')

        if label_names:
            x_pos = np.arange(len(label_names))
            width = 0.8 / len(model_names)

            for i, model_name in enumerate(model_names):
                per_class_f1 = self.model_results[model_name]['test_results']['per_class_metrics']['f1_score']
                offset = (i - len(model_names)/2) * width + width/2
                ax5.bar(x_pos + offset, per_class_f1, width,
                       label=model_name.upper(), alpha=0.7)

            ax5.set_xlabel('Class')
            ax5.set_ylabel('F1-Score')
            ax5.set_title('Per-Class F1-Score Comparison')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(label_names, rotation=45, ha='right')
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')

        # 6. Model efficiency (accuracy per parameter)
        ax6 = plt.subplot(2, 3, 6)
        efficiency = [acc / (params / 1000) for acc, params in zip(accuracies, num_params)]

        bars = ax6.bar(model_names, efficiency, color=colors, alpha=0.7)
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Accuracy per 1K Parameters')
        ax6.set_title('Model Efficiency')
        ax6.set_xticklabels([m.upper() for m in model_names], rotation=45, ha='right')
        ax6.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nComparison plot saved to: {os.path.join(self.results_dir, 'model_comparison.png')}")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models side by side"""
        if not self.model_results:
            return

        n_models = len(self.model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, results) in zip(axes, self.model_results.items()):
            cm = np.array(results['test_results']['confusion_matrix'])
            label_names = results.get('label_names')

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=label_names, yticklabels=label_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{model_name.upper()}\nAcc: {results["test_results"]["accuracy"]:.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrices_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrices saved to: {os.path.join(self.results_dir, 'confusion_matrices_comparison.png')}")

    def generate_report(self):
        """Generate comprehensive comparison report"""
        if not self.model_results:
            print("No results to compare")
            return

        # Generate comparison table
        df = self.generate_comparison_table()

        # Save table to CSV
        csv_path = os.path.join(self.results_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)

        # Print table
        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

        # Generate plots
        self.plot_comparison()
        self.plot_confusion_matrices()

        # Save detailed report as JSON
        report = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'num_models': len(self.model_results),
            'model_names': list(self.model_results.keys()),
            'comparison_table': df.to_dict('records'),
            'detailed_results': self.model_results
        }

        json_path = os.path.join(self.results_dir, 'detailed_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\nDetailed results saved to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")

        # Identify best model
        best_model = df.iloc[0]['Model']
        best_accuracy = df.iloc[0]['Accuracy']

        print(f"\nBest Model: {best_model} (Accuracy: {best_accuracy:.4f})")

        return df


def compare_all_models(data_dir='./tactile_data', batch_size=32,
                       num_epochs=100, learning_rate=0.001):
    """
    Train and compare all available models

    Args:
        data_dir: directory containing training data
        batch_size: batch size for training
        num_epochs: number of training epochs
        learning_rate: learning rate

    Returns:
        comparison DataFrame
    """
    # List of models to compare
    models = ['mlp', 'cnn', 'resnet', 'deepcnn', 'attention']

    # Create comparator
    comparator = ModelComparator(results_dir='./comparison_results')

    # Train all models
    comparator.train_all_models(
        model_names=models,
        data_dir=data_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    # Generate report
    comparison_df = comparator.generate_report()

    return comparison_df


if __name__ == '__main__':
    # Compare all models
    comparison_df = compare_all_models(
        data_dir='./tactile_data',
        batch_size=32,
        num_epochs=100,
        learning_rate=0.001
    )
