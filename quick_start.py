"""
Quick start script for tactile sensor shape classification
Simple interface for common use cases
"""
import os
import sys
import json
from config_utils import load_config


def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("TACTILE SENSOR SHAPE CLASSIFICATION - QUICK START")
    print("="*70)
    print("\n1. Collect training data from sensor")
    print("2. Collect evaluation data from sensor")
    print("3. Explore existing data")
    print("4. Train a single model")
    print("5. Train and compare all models")
    print("6. Evaluate model online (real-time predictions)")
    print("7. Exit")
    print("\n" + "="*70)


def collect_data(is_eval=False):
    """Collect data from sensor

    Args:
        is_eval: If True, collect evaluation data; otherwise collect training data
    """
    dataset_type = "EVALUATION" if is_eval else "TRAINING"
    print(f"\nStarting {dataset_type} data collection...")
    print("Make sure your tactile sensor is connected!")

    # Load config
    try:
        config = load_config()
        data_config = config.get_data_config()
        sensor_config = config.get_sensor_config()
        paths_config = config.get_paths_config()
    except:
        print("Warning: Could not load config, using defaults")
        data_config = {"shape_labels": ["sphere", "cube", "cylinder"], "samples_per_shape": 100}
        sensor_config = {"port": "/dev/ttyUSB0", "baud_rate": 2000000}
        paths_config = {"data_dir": "./tactile_data", "eval_data_dir": "./eval_data"}

    # Get user input with config defaults
    default_shapes = ",".join(data_config.get('shape_labels', ["sphere", "cube", "cylinder"]))
    shapes = input(f"\nEnter shape labels (comma-separated, default: {default_shapes}): ").strip()
    if not shapes:
        shapes = default_shapes
    shapes_list = [s.strip() for s in shapes.split(',')]

    default_samples = data_config.get('samples_per_shape', 100)
    samples = input(f"Samples per shape (default: {default_samples}): ").strip()
    if not samples:
        samples = str(default_samples)

    # Determine save directory
    if is_eval:
        save_dir = paths_config.get('eval_data_dir', './eval_data')
    else:
        save_dir = paths_config.get('data_dir', './tactile_data')

    print(f"\nWill collect {samples} samples for: {', '.join(shapes_list)}")
    print(f"Using sensor port: {sensor_config.get('port', '/dev/ttyUSB0')}")
    print(f"Save directory: {save_dir}")
    input("Press Enter to start...")

    from collect_data import TactileDataCollector

    # Update config with user choices
    config.set('data_collection.shape_labels', shapes_list)
    config.set('data_collection.samples_per_shape', int(samples))
    config.set('data_collection.data_dir', save_dir)

    collector = TactileDataCollector(config=config)

    try:
        collector.start_sensor()
        collector.collect_dataset(
            shape_labels=shapes_list,
            samples_per_shape=int(samples),
            save_dir=save_dir
        )
        print(f"\n✓ {dataset_type} data collection completed!")
        print(f"Data saved to: {save_dir}")
    finally:
        collector.close()


def explore_data():
    """Explore existing data"""
    print("\nExploring data...")

    from dataset import TactileDataLoader

    loader = TactileDataLoader(data_dir='./tactile_data')
    stats = loader.get_data_statistics()

    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Data shape: {stats['data_shape']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Class names: {stats['class_names']}")
    print(f"\nClass distribution:")
    for class_name, count in stats['class_counts'].items():
        print(f"  {class_name}: {count}")
    print(f"\nData range:")
    print(f"  Min: {stats['data_min']:.4f}")
    print(f"  Max: {stats['data_max']:.4f}")
    print(f"  Mean: {stats['data_mean']:.4f}")
    print(f"  Std: {stats['data_std']:.4f}")


def train_single_model():
    """Train a single model"""
    # Load config
    try:
        config = load_config()
        training_config = config.get_training_config()
        models_config = config.get_model_config()
        paths_config = config.get_paths_config()
    except:
        print("Warning: Could not load config, using defaults")
        training_config = {"num_epochs": 50, "batch_size": 32, "learning_rate": 0.001}
        models_config = {"available_models": ["mlp", "cnn", "resnet", "attention"]}
        paths_config = {"eval_data_dir": "./eval_data"}

    print("\nAvailable models:")
    available_models = models_config.get('available_models', ["mlp", "cnn", "resnet", "attention"])
    for i, model in enumerate(available_models[:4], 1):  # Show first 4
        model_desc = {
            'mlp': 'MLP (Baseline)',
            'cnn': 'CNN (Standard)',
            'resnet': 'ResNet (Residual)',
            'attention': 'Attention (Attention mechanism)',
            'deepcnn': 'DeepCNN (Deep)'
        }
        print(f"  {i}. {model_desc.get(model, model.upper())}")

    model_map = {str(i+1): model for i, model in enumerate(available_models[:4])}

    default_choice = '2'  # CNN
    choice = input(f"\nSelect model (1-4, default: {default_choice}): ").strip()
    if not choice:
        choice = default_choice

    model_name = model_map.get(choice, 'cnn')

    # Get training parameters with config defaults
    default_epochs = training_config.get('num_epochs', 50)
    epochs = input(f"Number of epochs (default: {default_epochs}): ").strip()
    if not epochs:
        epochs = str(default_epochs)

    default_batch = training_config.get('batch_size', 32)
    batch_size = input(f"Batch size (default: {default_batch}): ").strip()
    if not batch_size:
        batch_size = str(default_batch)

    # Ask about using eval dataset
    eval_data_dir = paths_config.get('eval_data_dir', './eval_data')
    use_eval = 'n'
    if os.path.exists(eval_data_dir):
        use_eval = input(f"\nUse separate evaluation dataset from {eval_data_dir}? (y/n, default: y): ").strip().lower()
        if not use_eval:
            use_eval = 'y'

    # Ask about wandb - use config default
    wandb_enabled = config.get('wandb.enabled', False)
    default_wandb = 'y' if wandb_enabled else 'n'
    use_wandb = input(f"\nEnable Wandb logging? (y/n, default: {default_wandb}): ").strip().lower()
    if not use_wandb:
        use_wandb = default_wandb

    print(f"\nTraining {model_name.upper()} for {epochs} epochs...")
    if use_eval == 'y':
        print(f"Using evaluation dataset from: {eval_data_dir}")
    if use_wandb == 'y':
        print("Wandb logging enabled")

    from train import train_model

    # Update config with user choices
    config.set('training.num_epochs', int(epochs))
    config.set('training.batch_size', int(batch_size))

    results = train_model(
        model_name=model_name,
        config=config,
        num_epochs=int(epochs),
        batch_size=int(batch_size),
        eval_data_dir=eval_data_dir if use_eval == 'y' else None,
        use_wandb=(use_wandb == 'y')
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
    print(f"Test F1-Score: {results['test_results']['f1_score']:.4f}")
    print(f"\nResults saved to: ./results/{model_name}/")


def train_all_models():
    """Train and compare all models"""
    # Load config
    try:
        config = load_config()
        training_config = config.get_training_config()
        paths_config = config.get_paths_config()
    except:
        print("Warning: Could not load config, using defaults")
        training_config = {"num_epochs": 50, "batch_size": 32}
        paths_config = {"eval_data_dir": "./eval_data"}

    default_epochs = training_config.get('num_epochs', 50)
    epochs = input(f"Number of epochs per model (default: {default_epochs}): ").strip()
    if not epochs:
        epochs = str(default_epochs)

    # Ask about eval dataset
    eval_data_dir = paths_config.get('eval_data_dir', './eval_data')
    use_eval = 'n'
    if os.path.exists(eval_data_dir):
        use_eval = input(f"\nUse separate evaluation dataset from {eval_data_dir}? (y/n, default: y): ").strip().lower()
        if not use_eval:
            use_eval = 'y'

    # Ask about wandb
    wandb_enabled = config.get('wandb.enabled', False)
    default_wandb = 'y' if wandb_enabled else 'n'
    use_wandb = input(f"\nEnable Wandb logging? (y/n, default: {default_wandb}): ").strip().lower()
    if not use_wandb:
        use_wandb = default_wandb

    print(f"\nTraining all models for {epochs} epochs each...")
    if use_eval == 'y':
        print(f"Using evaluation dataset from: {eval_data_dir}")
    if use_wandb == 'y':
        print("Wandb logging enabled")
    print("This may take a while!")

    from compare_models import compare_all_models

    comparison_df = compare_all_models(
        data_dir='./tactile_data',
        eval_data_dir=eval_data_dir if use_eval == 'y' else None,
        batch_size=32,
        num_epochs=int(epochs),
        learning_rate=0.001,
        use_wandb=(use_wandb == 'y')
    )

    print("\n" + "="*70)
    print("ALL MODELS TRAINED!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Individual models: ./results/")
    print("  - Comparison: ./comparison_results/")


def evaluate_online():
    """Evaluate model online with real-time predictions"""
    print("\nOnline Model Evaluation")
    print("="*50)
    
    # Load config
    try:
        config = load_config()
        eval_config = config.get_evaluation_config()
        sensor_config = config.get_sensor_config()
        paths_config = config.get_paths_config()
    except:
        print("Warning: Could not load config, using defaults")
        eval_config = {"min_confidence": 0.5, "smooth_predictions": True}
        sensor_config = {"port": "/dev/ttyUSB0"}
        paths_config = {"results_dir": "./results"}
    
    # Check for trained models in both possible locations
    results_dir = paths_config.get('results_dir', './results')
    comparison_dir = paths_config.get('comparison_dir', './comparison_results')
    
    models_found = False
    search_dir = results_dir
    
    if os.path.exists(results_dir) and any(os.path.isdir(os.path.join(results_dir, d)) for d in os.listdir(results_dir)):
        models_found = True
        search_dir = results_dir
    elif os.path.exists(comparison_dir) and any(os.path.isdir(os.path.join(comparison_dir, d)) for d in os.listdir(comparison_dir)):
        models_found = True
        search_dir = comparison_dir
        print(f"📁 Using models from comparison results: {comparison_dir}")
    
    if not models_found:
        print("❌ No trained models found!")
        print("Please train a model first using option 3 or 4.")
        return
    
    print("\nThis will connect to your tactile sensor and make real-time predictions.")

    # Use config values directly (no prompts)
    port = sensor_config.get('port', '/dev/ttyUSB0')
    min_confidence = eval_config.get('min_confidence', 0.5)
    smooth_predictions = eval_config.get('smooth_predictions', True)

    print(f"Using sensor port: {port}")
    print(f"Minimum confidence: {min_confidence}")
    print(f"Smooth predictions: {'Yes' if smooth_predictions else 'No'}")

    from eval_online import TactileOnlineEvaluator

    try:
        # List all available models
        available_models = []
        for model_dir in os.listdir(search_dir):
            model_path_dir = os.path.join(search_dir, model_dir)
            if os.path.isdir(model_path_dir):
                results_file = os.path.join(model_path_dir, 'results.json')
                model_file = os.path.join(model_path_dir, 'best_model.pth')

                if os.path.exists(results_file) and os.path.exists(model_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                    accuracy = results.get('test_results', {}).get('accuracy', 0)
                    base_model_name = model_dir.split('_')[0]
                    available_models.append({
                        'name': base_model_name,
                        'full_name': model_dir,
                        'path': model_file,
                        'accuracy': accuracy
                    })

        if not available_models:
            print("❌ No valid trained models found")
            return

        # Sort by accuracy (best first)
        available_models.sort(key=lambda x: x['accuracy'], reverse=True)

        # Display available models
        print("\n📋 Available models:")
        for idx, model in enumerate(available_models, 1):
            print(f"  {idx}. {model['full_name']} (accuracy: {model['accuracy']:.3f})")

        # Let user choose model
        while True:
            choice = input(f"\nSelect model (1-{len(available_models)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("❌ Please enter a valid number")

        print(f"\n🎯 Using model: {selected_model['full_name']} (accuracy: {selected_model['accuracy']:.3f})")
        model_name = selected_model['name']
        model_path = selected_model['path']

        print(f"\nStarting online evaluation...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save prediction history")

        input("\nPress Enter when ready...")
        
        # Create evaluator
        evaluator = TactileOnlineEvaluator(
            model_path=model_path,
            model_name=model_name,
            config=config
        )
        
        # Start sensor
        evaluator.start_sensor()
        
        # Run evaluation
        evaluator.run_evaluation(
            min_confidence=min_confidence,
            smooth_predictions=smooth_predictions
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check sensor connection")
        print("  2. Verify port (try ls /dev/ttyUSB*)")
        print("  3. Check sensor permissions (sudo chmod 666 /dev/ttyUSB0)")


def main():
    """Main quick start interface"""
    while True:
        print_menu()
        choice = input("\nSelect option (1-7): ").strip()

        try:
            if choice == '1':
                collect_data(is_eval=False)
            elif choice == '2':
                collect_data(is_eval=True)
            elif choice == '3':
                explore_data()
            elif choice == '4':
                train_single_model()
            elif choice == '5':
                train_all_models()
            elif choice == '6':
                evaluate_online()
            elif choice == '7':
                print("\nExiting... Goodbye!")
                sys.exit(0)
            else:
                print("\nInvalid choice! Please select 1-7.")
                continue

            input("\nPress Enter to return to menu...")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            input("Press Enter to return to menu...")
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter to return to menu...")


if __name__ == '__main__':
    main()
