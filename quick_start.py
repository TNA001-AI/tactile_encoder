"""
Quick start script for tactile sensor shape classification
Simple interface for common use cases
"""
import os
import sys


def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("TACTILE SENSOR SHAPE CLASSIFICATION - QUICK START")
    print("="*70)
    print("\n1. Collect new data from sensor")
    print("2. Explore existing data")
    print("3. Train a single model")
    print("4. Train and compare all models")
    print("5. Run full pipeline (with existing data)")
    print("6. Run full pipeline (collect new data)")
    print("7. Create custom configuration file")
    print("8. Exit")
    print("\n" + "="*70)


def collect_data():
    """Collect data from sensor"""
    print("\nStarting data collection...")
    print("Make sure your tactile sensor is connected!")

    shapes = input("\nEnter shape labels (comma-separated, default: sphere,cube,cylinder): ").strip()
    if not shapes:
        shapes = "sphere,cube,cylinder"
    shapes_list = [s.strip() for s in shapes.split(',')]

    samples = input("Samples per shape (default: 100): ").strip()
    if not samples:
        samples = "100"

    print(f"\nWill collect {samples} samples for: {', '.join(shapes_list)}")
    input("Press Enter to start...")

    from collect_data import TactileDataCollector

    collector = TactileDataCollector(port='/dev/ttyUSB0', baud=2000000)

    try:
        collector.start_sensor()
        collector.collect_dataset(
            shape_labels=shapes_list,
            samples_per_shape=int(samples),
            save_dir='./tactile_data'
        )
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
    print("\nAvailable models:")
    print("  1. MLP (Baseline)")
    print("  2. CNN (Standard)")
    print("  3. ResNet (Residual)")
    print("  4. DeepCNN (Deeper)")
    print("  5. Attention (Attention mechanism)")

    model_map = {
        '1': 'mlp',
        '2': 'cnn',
        '3': 'resnet',
        '4': 'deepcnn',
        '5': 'attention'
    }

    choice = input("\nSelect model (1-5, default: 2-CNN): ").strip()
    if not choice:
        choice = '2'

    model_name = model_map.get(choice, 'cnn')

    epochs = input("Number of epochs (default: 50): ").strip()
    if not epochs:
        epochs = "50"

    batch_size = input("Batch size (default: 32): ").strip()
    if not batch_size:
        batch_size = "32"

    print(f"\nTraining {model_name.upper()} for {epochs} epochs...")

    from train import train_model

    results = train_model(
        model_name=model_name,
        data_dir='./tactile_data',
        batch_size=int(batch_size),
        num_epochs=int(epochs),
        learning_rate=0.001
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
    print(f"Test F1-Score: {results['test_results']['f1_score']:.4f}")
    print(f"\nResults saved to: ./results/{model_name}/")


def train_all_models():
    """Train and compare all models"""
    epochs = input("Number of epochs per model (default: 50): ").strip()
    if not epochs:
        epochs = "50"

    print(f"\nTraining all 5 models for {epochs} epochs each...")
    print("This may take a while!")

    from compare_models import compare_all_models

    comparison_df = compare_all_models(
        data_dir='./tactile_data',
        batch_size=32,
        num_epochs=int(epochs),
        learning_rate=0.001
    )

    print("\n" + "="*70)
    print("ALL MODELS TRAINED!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Individual models: ./results/")
    print("  - Comparison: ./comparison_results/")


def run_full_pipeline(skip_collection=True):
    """Run full pipeline"""
    if not skip_collection:
        print("\nFull pipeline will:")
        print("  1. Collect data from sensor")
        print("  2. Explore data")
        print("  3. Train all models")
        print("  4. Compare models")
    else:
        print("\nPipeline will:")
        print("  1. Explore existing data")
        print("  2. Train all models")
        print("  3. Compare models")

    epochs = input("\nNumber of epochs per model (default: 50): ").strip()
    if not epochs:
        epochs = "50"

    print("\nStarting pipeline...")

    from pipeline import TactileClassificationPipeline

    config = TactileClassificationPipeline.get_default_config()
    config['num_epochs'] = int(epochs)

    pipeline = TactileClassificationPipeline(config=config)
    pipeline.run_full_pipeline(skip_collection=skip_collection)


def create_config():
    """Create custom configuration file"""
    print("\nCreating custom configuration...")

    config_name = input("Configuration name (default: my_config): ").strip()
    if not config_name:
        config_name = "my_config"

    shapes = input("Shape labels (comma-separated, default: sphere,cube,cylinder): ").strip()
    if not shapes:
        shapes = "sphere,cube,cylinder"
    shapes_list = [s.strip() for s in shapes.split(',')]

    samples = input("Samples per shape (default: 100): ").strip()
    if not samples:
        samples = "100"

    epochs = input("Training epochs (default: 100): ").strip()
    if not epochs:
        epochs = "100"

    from pipeline import TactileClassificationPipeline
    import json

    config = TactileClassificationPipeline.get_default_config()
    config['shape_labels'] = shapes_list
    config['samples_per_shape'] = int(samples)
    config['num_epochs'] = int(epochs)

    filename = f"{config_name}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nConfiguration saved to: {filename}")
    print(f"\nTo use this configuration, run:")
    print(f"  python pipeline.py --config {filename} --step all")


def main():
    """Main quick start interface"""
    while True:
        print_menu()
        choice = input("\nSelect option (1-8): ").strip()

        try:
            if choice == '1':
                collect_data()
            elif choice == '2':
                explore_data()
            elif choice == '3':
                train_single_model()
            elif choice == '4':
                train_all_models()
            elif choice == '5':
                run_full_pipeline(skip_collection=True)
            elif choice == '6':
                run_full_pipeline(skip_collection=False)
            elif choice == '7':
                create_config()
            elif choice == '8':
                print("\nExiting... Goodbye!")
                sys.exit(0)
            else:
                print("\nInvalid choice! Please select 1-8.")
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
