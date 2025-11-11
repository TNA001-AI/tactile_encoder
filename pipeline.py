"""
Main pipeline orchestrator for tactile sensor shape classification
Handles complete workflow: data collection -> training -> evaluation -> comparison
"""
import os
import argparse
import json
from datetime import datetime

from collect_data import TactileDataCollector
from dataset import TactileDataLoader
from train import train_model
from compare_models import ModelComparator, compare_all_models


class TactileClassificationPipeline:
    """Complete pipeline for tactile shape classification"""

    def __init__(self, config=None):
        """
        Args:
            config: dict or path to config file
        """
        if config is None:
            self.config = self.get_default_config()
        elif isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        self.collector = None
        self.results = {}

    @staticmethod
    def get_default_config():
        """Get default configuration"""
        return {
            # Data collection
            'sensor_port': '/dev/ttyUSB0',
            'sensor_baud': 2000000,
            'sensor_shape': [16, 32],
            'shape_labels': ['sphere', 'cube', 'cylinder', 'cone', 'pyramid'],
            'samples_per_shape': 100,
            'data_dir': './tactile_data',

            # Training
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'test_size': 0.2,
            'val_size': 0.1,

            # Models
            'models_to_train': ['mlp', 'cnn', 'resnet', 'deepcnn', 'attention'],

            # Paths
            'results_dir': './results',
            'comparison_dir': './comparison_results'
        }

    def save_config(self, filepath):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to: {filepath}")

    def step1_collect_data(self):
        """Step 1: Collect tactile sensor data"""
        print("\n" + "="*80)
        print("STEP 1: DATA COLLECTION")
        print("="*80)

        self.collector = TactileDataCollector(
            port=self.config['sensor_port'],
            baud=self.config['sensor_baud'],
            sensor_shape=tuple(self.config['sensor_shape'])
        )

        try:
            self.collector.start_sensor()

            self.collector.collect_dataset(
                shape_labels=self.config['shape_labels'],
                samples_per_shape=self.config['samples_per_shape'],
                save_dir=self.config['data_dir']
            )

            print("\nData collection completed successfully!")

        except Exception as e:
            print(f"Error during data collection: {e}")
            raise
        finally:
            if self.collector:
                self.collector.close()

    def step2_explore_data(self):
        """Step 2: Explore collected data"""
        print("\n" + "="*80)
        print("STEP 2: DATA EXPLORATION")
        print("="*80)

        loader = TactileDataLoader(data_dir=self.config['data_dir'])

        try:
            stats = loader.get_data_statistics()

            print("\nDataset Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Data shape: {stats['data_shape']}")
            print(f"  Number of classes: {stats['num_classes']}")
            print(f"  Class names: {stats['class_names']}")
            print(f"\nClass distribution:")
            for class_name, count in stats['class_counts'].items():
                print(f"    {class_name}: {count}")
            print(f"\nData statistics:")
            print(f"  Min: {stats['data_min']:.4f}")
            print(f"  Max: {stats['data_max']:.4f}")
            print(f"  Mean: {stats['data_mean']:.4f}")
            print(f"  Std: {stats['data_std']:.4f}")

            self.results['data_stats'] = stats

        except Exception as e:
            print(f"Error during data exploration: {e}")
            print("Make sure you have collected data first (run step1_collect_data)")
            raise

    def step3_train_models(self, model_names=None):
        """
        Step 3: Train specified models

        Args:
            model_names: list of model names to train, or None to train all
        """
        print("\n" + "="*80)
        print("STEP 3: MODEL TRAINING")
        print("="*80)

        if model_names is None:
            model_names = self.config['models_to_train']

        self.results['trained_models'] = {}

        for i, model_name in enumerate(model_names):
            print(f"\n[{i+1}/{len(model_names)}] Training {model_name.upper()}...")

            try:
                results = train_model(
                    model_name=model_name,
                    data_dir=self.config['data_dir'],
                    save_dir=os.path.join(self.config['results_dir'], model_name),
                    batch_size=self.config['batch_size'],
                    num_epochs=self.config['num_epochs'],
                    learning_rate=self.config['learning_rate']
                )

                self.results['trained_models'][model_name] = results

                print(f"\n{model_name.upper()} training completed!")
                print(f"  Test Accuracy: {results['test_results']['accuracy']:.4f}")
                print(f"  Test F1-Score: {results['test_results']['f1_score']:.4f}")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        print(f"\n{'='*80}")
        print(f"Completed training {len(self.results['trained_models'])}/{len(model_names)} models")
        print(f"{'='*80}")

    def step4_compare_models(self):
        """Step 4: Compare all trained models"""
        print("\n" + "="*80)
        print("STEP 4: MODEL COMPARISON")
        print("="*80)

        comparator = ModelComparator(results_dir=self.config['comparison_dir'])

        # Load results from trained models
        model_dirs = {}
        for model_name in self.config['models_to_train']:
            model_dir = os.path.join(self.config['results_dir'], model_name)
            if os.path.exists(model_dir):
                model_dirs[model_name] = model_dir

        if not model_dirs:
            print("No trained models found. Please run step3_train_models first.")
            return

        comparator.load_results_from_files(model_dirs)
        comparison_df = comparator.generate_report()

        self.results['comparison'] = comparison_df.to_dict('records')

        print("\nModel comparison completed!")

    def run_full_pipeline(self, skip_collection=False):
        """
        Run the complete pipeline

        Args:
            skip_collection: if True, skip data collection step
        """
        print("\n" + "="*80)
        print("TACTILE SENSOR SHAPE CLASSIFICATION PIPELINE")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save configuration
        config_file = os.path.join(self.config['results_dir'], f'config_{timestamp}.json')
        os.makedirs(self.config['results_dir'], exist_ok=True)
        self.save_config(config_file)

        try:
            # Step 1: Collect data (optional)
            if not skip_collection:
                self.step1_collect_data()

            # Step 2: Explore data
            self.step2_explore_data()

            # Step 3: Train models
            self.step3_train_models()

            # Step 4: Compare models
            self.step4_compare_models()

            # Save pipeline results
            results_file = os.path.join(self.config['results_dir'], f'pipeline_results_{timestamp}.json')
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=4)

            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\nResults saved to:")
            print(f"  - Configuration: {config_file}")
            print(f"  - Pipeline results: {results_file}")
            print(f"  - Model results: {self.config['results_dir']}")
            print(f"  - Comparison: {self.config['comparison_dir']}")

        except Exception as e:
            print(f"\nPipeline error: {e}")
            raise


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='Tactile Sensor Shape Classification Pipeline'
    )

    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['collect', 'explore', 'train', 'compare', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection (use existing data)')
    parser.add_argument('--models', nargs='+',
                       choices=['mlp', 'cnn', 'resnet', 'deepcnn', 'vgg', 'attention'],
                       help='Specific models to train')
    parser.add_argument('--data-dir', type=str, default='./tactile_data',
                       help='Directory containing tactile data')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')

    args = parser.parse_args()

    # Create pipeline
    if args.config:
        pipeline = TactileClassificationPipeline(config=args.config)
    else:
        config = TactileClassificationPipeline.get_default_config()

        # Override with command line arguments
        config['data_dir'] = args.data_dir
        config['results_dir'] = args.results_dir
        config['batch_size'] = args.batch_size
        config['num_epochs'] = args.epochs
        config['learning_rate'] = args.learning_rate

        if args.models:
            config['models_to_train'] = args.models

        pipeline = TactileClassificationPipeline(config=config)

    # Run specified step
    if args.step == 'collect':
        pipeline.step1_collect_data()
    elif args.step == 'explore':
        pipeline.step2_explore_data()
    elif args.step == 'train':
        pipeline.step3_train_models()
    elif args.step == 'compare':
        pipeline.step4_compare_models()
    elif args.step == 'all':
        pipeline.run_full_pipeline(skip_collection=args.skip_collection)


if __name__ == '__main__':
    main()
