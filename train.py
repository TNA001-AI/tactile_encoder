"""
Training pipeline for tactile sensor shape classification models
"""
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from dataset import TactileDataLoader
from models import get_model
from config_utils import load_config, override_config_from_args

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class Trainer:
    """Training pipeline for tactile classification models"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_classes, label_names, config=None, save_dir=None, use_wandb=False, wandb_project=None):
        """
        Args:
            model: PyTorch model
            train_loader: training data loader
            val_loader: validation data loader
            test_loader: test data loader
            num_classes: number of output classes
            label_names: list of class names
            config: ConfigManager instance
            save_dir: directory to save results (overrides config)
            use_wandb: whether to use wandb for logging
            wandb_project: wandb project name
        """
        if config is None:
            from config_utils import load_config
            config = load_config()

        self.config = config
        training_config = config.get_training_config()

        # Device setup
        if training_config.get('use_cuda', True) and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.label_names = label_names

        # Save directory
        if save_dir is None:
            save_dir = config.get('paths.results_dir', './results')
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without wandb.")
            self.use_wandb = False

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate(self, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.model(data)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        return val_loss, val_acc

    def train(self, num_epochs=None, learning_rate=None, weight_decay=None,
              optimizer_name='adam', scheduler_name='plateau', early_stopping_patience=None):
        """
        Train the model

        Args:
            num_epochs: number of training epochs (uses config if None)
            learning_rate: initial learning rate (uses config if None)
            weight_decay: L2 regularization (uses config if None)
            optimizer_name: 'adam' or 'sgd'
            scheduler_name: 'plateau' or 'cosine'
            early_stopping_patience: epochs to wait before early stopping (uses config if None)
        """
        # Get training config
        training_config = self.config.get_training_config()
        
        # Use config values if not provided
        if num_epochs is None:
            num_epochs = training_config.get('num_epochs', 100)
        if learning_rate is None:
            learning_rate = training_config.get('learning_rate', 0.001)
        if weight_decay is None:
            weight_decay = training_config.get('weight_decay', 1e-4)
        if early_stopping_patience is None:
            early_stopping_patience = training_config.get('patience', 15)
        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                 momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")

        # Learning rate scheduler
        if scheduler_name.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                         patience=5)
        elif scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            scheduler = None

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 60)

        start_time = time.time()
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)

            # Validate
            val_loss, val_acc = self.validate(criterion)

            epoch_time = time.time() - epoch_start

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)

            # Learning rate scheduling
            if scheduler_name.lower() == 'plateau':
                scheduler.step(val_acc)
            elif scheduler_name.lower() == 'cosine':
                scheduler.step()

            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'epoch_time': epoch_time,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

        # Load best model for testing
        self.load_checkpoint('best_model.pth')

        return self.history

    def test(self):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = \
            precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist(),
                'support': support.tolist()
            }
        }

        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print("=" * 60)

        # Log test results to wandb
        if self.use_wandb:
            wandb.log({
                'test/accuracy': accuracy,
                'test/precision': precision,
                'test/recall': recall,
                'test/f1_score': f1
            })

            # Log confusion matrix as image
            import matplotlib
            matplotlib.use('Agg')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_names,
                       yticklabels=self.label_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            wandb.log({"test/confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

        return results, all_preds, all_labels, all_probs

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()

    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'),
                       dpi=300, bbox_inches='tight')

        plt.close()


def train_model(model_name, config=None, data_dir=None, eval_data_dir=None, save_dir=None,
                batch_size=None, num_epochs=None, learning_rate=None, use_wandb=False, wandb_project=None):
    """
    Complete training pipeline for a single model

    Args:
        model_name: name of the model to train
        config: ConfigManager instance (loads default if None)
        data_dir: directory containing training data (overrides config)
        eval_data_dir: directory containing evaluation data (overrides config)
        save_dir: directory to save results (overrides config)
        batch_size: batch size for training (overrides config)
        num_epochs: number of training epochs (overrides config)
        learning_rate: learning rate (overrides config)
        use_wandb: whether to use wandb for logging
        wandb_project: wandb project name

    Returns:
        results dictionary
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Get config values with overrides
    training_config = config.get_training_config()
    paths_config = config.get_paths_config()

    if data_dir is None:
        data_dir = paths_config.get('data_dir', './tactile_data')
    if eval_data_dir is None:
        eval_data_dir = paths_config.get('eval_data_dir', None)
    if save_dir is None:
        save_dir = os.path.join(paths_config.get('results_dir', './results'), model_name)
    if batch_size is None:
        batch_size = training_config.get('batch_size', 32)
    if num_epochs is None:
        num_epochs = training_config.get('num_epochs', 100)
    if learning_rate is None:
        learning_rate = training_config.get('learning_rate', 0.001)

    # Wandb config
    if use_wandb is None:
        use_wandb = config.get('wandb.enabled', False)
    if wandb_project is None:
        wandb_project = config.get('wandb.project', 'tactile-classification')
    # Setup
    device = torch.device('cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(save_dir, exist_ok=True)

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=f"{model_name}_{timestamp}",
            config={
                "model": model_name,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "data_dir": data_dir,
                "eval_data_dir": eval_data_dir
            }
        )

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} model")
    print(f"{'='*60}")
    if use_wandb and WANDB_AVAILABLE:
        print(f"Wandb logging enabled - Project: {wandb_project}")

    # Load data
    data_loader = TactileDataLoader(data_dir=data_dir, eval_data_dir=eval_data_dir)
    train_loader, val_loader, test_loader, num_classes, label_names = \
        data_loader.prepare_dataloaders(batch_size=batch_size)

    # Get input shape from first batch
    sample_data, _ = next(iter(train_loader))
    input_shape = tuple(sample_data.shape[1:])

    # Create model
    model = get_model(model_name, input_shape=input_shape, num_classes=num_classes)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader,
                     num_classes, label_names, config=config, save_dir=save_dir,
                     use_wandb=use_wandb, wandb_project=wandb_project)

    # Train
    history = trainer.train(num_epochs=num_epochs, learning_rate=learning_rate)

    # Test
    test_results, preds, labels, probs = trainer.test()

    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(np.array(test_results['confusion_matrix']))

    # Save results
    results = {
        'model_name': model_name,
        'num_parameters': num_params,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'label_names': label_names.tolist() if label_names is not None else None,
        'training_history': history,
        'test_results': test_results,
        'timestamp': timestamp
    }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {save_dir}")

    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return results


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Train tactile classification model')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--model', type=str, help='Model name (overrides config)')
    parser.add_argument('--data-dir', type=str, help='Training data directory (overrides config)')
    parser.add_argument('--eval-data-dir', type=str, help='Evaluation data directory (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--save-dir', type=str, help='Results directory (overrides config)')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, help='Wandb project name')

    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("Using default configuration...")
        from config_utils import get_default_config
        config_dict = get_default_config()
        from config_utils import ConfigManager
        config = ConfigManager.__new__(ConfigManager)
        config.config = config_dict
    
    # Override with command line arguments
    config = override_config_from_args(config, args)
    
    # Get model name
    model_name = args.model or config.get('models.default_model', 'cnn')
    
    print("\n" + "="*70)
    print("TACTILE MODEL TRAINING")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Data directory: {config.get('paths.data_dir', './tactile_data')}")
    print(f"Batch size: {config.get('training.batch_size', 32)}")
    print(f"Epochs: {config.get('training.num_epochs', 100)}")
    print(f"Learning rate: {config.get('training.learning_rate', 0.001)}")
    print("="*70)
    
    # Train model
    try:
        results = train_model(
            model_name=model_name,
            config=config,
            data_dir=args.data_dir,
            eval_data_dir=args.eval_data_dir,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project
        )

        print(f"\n✓ Training completed successfully!")
        print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"Results saved to: {results.get('save_dir', 'results')}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if WANDB_AVAILABLE and args.wandb:
            wandb.finish()
        raise


if __name__ == '__main__':
    main()
