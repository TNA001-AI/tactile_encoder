"""
Configuration utilities for tactile sensor shape classification
Handles loading and merging of configuration files
"""
import json
import os
import copy
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize config manager
        
        Args:
            config_path: Path to main config file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = ['sensor', 'data_collection', 'training', 'models', 'paths']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
            
        except Exception as e:
            raise Exception(f"Failed to load config from {config_path}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., 'sensor.port')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        target = self.config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set the final key
        target[keys[-1]] = value
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update config with values from dictionary (supports dot notation keys)
        
        Args:
            updates: Dictionary with updates
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file
        
        Args:
            output_path: Output file path (defaults to original path)
        """
        output_path = output_path or self.config_path
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def copy(self) -> 'ConfigManager':
        """Create a deep copy of the config manager"""
        new_manager = ConfigManager.__new__(ConfigManager)
        new_manager.config_path = self.config_path
        new_manager.config = copy.deepcopy(self.config)
        return new_manager
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """Get sensor-specific configuration"""
        return self.config['sensor']
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return self.config['training']
    
    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model-specific configuration
        
        Args:
            model_name: Specific model name, or None for general model config
        """
        model_config = self.config['models']
        if model_name and model_name in model_config.get('model_params', {}):
            # Merge general config with model-specific config
            general = {k: v for k, v in model_config.items() if k != 'model_params'}
            specific = model_config['model_params'][model_name]
            return {**general, **specific}
        return model_config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data collection configuration"""
        return self.config['data_collection']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config['evaluation']
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.config['paths']


def load_config(config_path: str = "config.json") -> ConfigManager:
    """
    Load configuration manager
    
    Args:
        config_path: Path to config file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def create_custom_config(base_config_path: str = "config.json", 
                        output_path: str = "custom_config.json",
                        **overrides) -> str:
    """
    Create custom configuration file with overrides
    
    Args:
        base_config_path: Base configuration file
        output_path: Output path for custom config
        **overrides: Configuration overrides (use dot notation for nested keys)
        
    Returns:
        Path to created config file
    """
    config = load_config(base_config_path)
    config.update_from_dict(overrides)
    config.save(output_path)
    return output_path


def get_default_config() -> Dict[str, Any]:
    """Get default configuration as dictionary"""
    try:
        config = load_config()
        return config.config
    except:
        # Fallback default config if file doesn't exist
        return {
            "sensor": {
                "port": "/dev/ttyUSB0",
                "baud_rate": 2000000,
                "shape": [16, 32],
                "timeout": 1
            },
            "data_collection": {
                "shape_labels": ["sphere", "cube", "cylinder", "cone", "pyramid"],
                "samples_per_shape": 100,
                "data_dir": "./tactile_data",
                "threshold": 30,
                "noise_scale": 50,
                "gaussian_sigma": 0.5
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "patience": 10,
                "min_delta": 0.001,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42,
                "use_cuda": True,
                "num_workers": 4
            },
            "models": {
                "available_models": ["mlp", "cnn", "resnet", "deepcnn", "attention"],
                "default_model": "cnn"
            },
            "paths": {
                "data_dir": "./tactile_data",
                "results_dir": "./results",
                "comparison_dir": "./comparison_results",
                "config_dir": "./configs"
            },
            "evaluation": {
                "min_confidence": 0.5,
                "smooth_predictions": True,
                "smoothing_window": 5
            }
        }


# Command line argument integration
def override_config_from_args(config: ConfigManager, args) -> ConfigManager:
    """
    Override config with command line arguments
    
    Args:
        config: ConfigManager instance
        args: Parsed command line arguments
        
    Returns:
        Updated ConfigManager
    """
    config = config.copy()
    
    # Map common command line args to config paths
    arg_mappings = {
        'data_dir': 'paths.data_dir',
        'results_dir': 'paths.results_dir',
        'batch_size': 'training.batch_size',
        'epochs': 'training.num_epochs',
        'learning_rate': 'training.learning_rate',
        'port': 'sensor.port',
        'model': 'models.default_model',
        'min_confidence': 'evaluation.min_confidence'
    }
    
    for arg_name, config_path in arg_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            config.set(config_path, getattr(args, arg_name))
    
    return config


if __name__ == '__main__':
    # Test configuration loading
    try:
        config = load_config()
        print("✓ Config loaded successfully")
        print(f"Sensor port: {config.get('sensor.port')}")
        print(f"Training epochs: {config.get('training.num_epochs')}")
        print(f"Available models: {config.get('models.available_models')}")
        
        # Test custom config creation
        custom_path = create_custom_config(
            output_path="test_config.json",
            **{
                "training.num_epochs": 50,
                "sensor.port": "/dev/ttyUSB1",
                "evaluation.min_confidence": 0.7
            }
        )
        print(f"✓ Custom config created: {custom_path}")
        
        # Cleanup test file
        os.remove(custom_path)
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")