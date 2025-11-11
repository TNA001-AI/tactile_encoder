# Tactile Sensor Shape Classification Pipeline

A complete machine learning pipeline for collecting tactile sensor data and training multiple deep learning models for shape classification tasks.

## Overview

This pipeline provides an end-to-end solution for:
1. **Data Collection**: Collect labeled tactile sensor data from a 16x32 tactile array
2. **Data Processing**: Load, preprocess, and split data for training
3. **Model Training**: Train multiple neural network architectures
4. **Model Comparison**: Compare and evaluate different models
5. **Visualization**: Generate comprehensive plots and reports

## Features

### Supported Models
- **MLP**: Multi-Layer Perceptron (fully connected baseline)
- **CNN**: Standard Convolutional Neural Network (3 conv layers)
- **ResNet**: Residual Network with skip connections
- **DeepCNN**: Deeper CNN with 4 conv layers and more filters
- **VGG**: VGG-style architecture with conv-conv-pool blocks
- **Attention**: CNN with spatial attention mechanism

### Key Capabilities
- Real-time tactile data visualization during collection
- Automatic train/validation/test split with stratification
- Early stopping and learning rate scheduling
- Comprehensive metrics (accuracy, precision, recall, F1-score)
- Confusion matrices and training curves
- Model comparison plots and tables
- Detailed JSON reports

## Installation

1. Clone this repository or navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
# Run complete pipeline (skip data collection if you already have data)
python pipeline.py --step all --skip-collection --data-dir ./tactile_data --epochs 50
```

### Option 2: Step-by-Step Execution

#### Step 1: Collect Data
```bash
# Collect data for different shapes
python collect_data.py
```

Or use the pipeline:
```bash
python pipeline.py --step collect
```

The data collector will:
- Connect to your tactile sensor (default: /dev/ttyUSB0)
- Initialize and calibrate the sensor
- Guide you through collecting samples for each shape
- Press 's' to save each sample, 'q' to finish early
- Save data as .npz files in `./tactile_data/`

#### Step 2: Explore Data
```bash
python pipeline.py --step explore --data-dir ./tactile_data
```

#### Step 3: Train Models
```bash
# Train all models
python pipeline.py --step train --data-dir ./tactile_data --epochs 100

# Train specific models
python pipeline.py --step train --models cnn resnet --epochs 50
```

Or train a single model:
```bash
python train.py
```

#### Step 4: Compare Models
```bash
python pipeline.py --step compare
```

Or:
```bash
python compare_models.py
```

## Usage Examples

### Example 1: Train Single Model
```python
from train import train_model

results = train_model(
    model_name='cnn',
    data_dir='./tactile_data',
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001
)
```

### Example 2: Compare All Models
```python
from compare_models import compare_all_models

comparison_df = compare_all_models(
    data_dir='./tactile_data',
    batch_size=32,
    num_epochs=100
)
```

### Example 3: Custom Pipeline
```python
from pipeline import TactileClassificationPipeline

# Create custom configuration
config = {
    'shape_labels': ['sphere', 'cube', 'cylinder'],
    'samples_per_shape': 150,
    'models_to_train': ['cnn', 'resnet'],
    'num_epochs': 50
}

pipeline = TactileClassificationPipeline(config=config)
pipeline.run_full_pipeline(skip_collection=True)
```

### Example 4: Load and Use Trained Model
```python
import torch
from models import get_model

# Load model
model = get_model('cnn', input_shape=(16, 32), num_classes=5)
checkpoint = torch.load('./results/cnn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(tactile_data)
```

## Project Structure

```
tactile_encoder/
├── collect_data.py          # Data collection from sensor
├── dataset.py               # Dataset and DataLoader utilities
├── models.py                # All model architectures
├── train.py                 # Training pipeline
├── compare_models.py        # Model comparison and evaluation
├── pipeline.py              # Main pipeline orchestrator
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── tactile_data/           # Collected sensor data (created during collection)
│   ├── sphere_*.npz
│   ├── cube_*.npz
│   └── dataset_*.npz
│
├── results/                # Training results (created during training)
│   ├── mlp/
│   │   ├── best_model.pth
│   │   ├── results.json
│   │   ├── training_history.png
│   │   └── confusion_matrix.png
│   ├── cnn/
│   └── ...
│
└── comparison_results/     # Model comparison results
    ├── model_comparison.png
    ├── model_comparison.csv
    └── detailed_comparison.json
```

## Configuration

You can create a custom configuration file:

```json
{
    "sensor_port": "/dev/ttyUSB0",
    "sensor_baud": 2000000,
    "sensor_shape": [16, 32],
    "shape_labels": ["sphere", "cube", "cylinder", "cone", "pyramid"],
    "samples_per_shape": 100,
    "data_dir": "./tactile_data",
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "models_to_train": ["mlp", "cnn", "resnet", "lstm", "bilstm", "cnn_lstm"]
}
```

Then run:
```bash
python pipeline.py --config my_config.json --step all
```

## Command Line Arguments

### pipeline.py
```
--config PATH           Path to configuration JSON file
--step STEP            Step to run: collect, explore, train, compare, all
--skip-collection      Skip data collection step
--models MODEL [...]   Specific models to train
--data-dir PATH        Directory containing tactile data
--results-dir PATH     Directory to save results
--batch-size INT       Batch size for training (default: 32)
--epochs INT           Number of training epochs (default: 100)
--learning-rate FLOAT  Learning rate (default: 0.001)
```

## Model Architectures

### MLP
- Fully connected layers: 512 -> 256 -> 128
- BatchNorm + ReLU + Dropout after each layer
- Good baseline, fast training

### CNN
- 3 convolutional layers: 32 -> 64 -> 128 channels
- MaxPooling after each conv layer
- 3 fully connected layers: 256 -> 128 -> num_classes
- Best for spatial pattern recognition

### ResNet
- Residual blocks with skip connections
- 3 stages with 2 blocks each
- Global average pooling
- Excellent for complex patterns

### LSTM
- Treats tactile data as sequential (rows as time steps)
- 2-layer LSTM with hidden_dim=128
- Good for capturing temporal/spatial dependencies

### BiLSTM
- Bidirectional LSTM processing
- Captures context in both directions
- Better than LSTM for spatial data

### CNN-LSTM
- Hybrid architecture
- CNN for feature extraction
- LSTM for sequential processing
- Combines strengths of both approaches

## Output Files

### Training Results (per model)
- `best_model.pth`: Best model checkpoint
- `results.json`: Detailed metrics and configuration
- `training_history.png`: Loss and accuracy curves
- `confusion_matrix.png`: Confusion matrix heatmap

### Comparison Results
- `model_comparison.png`: 6-panel comparison plot
  - Metrics comparison
  - Accuracy vs parameters
  - Training curves
  - Training time
  - Per-class F1-scores
  - Model efficiency
- `model_comparison.csv`: Summary table
- `confusion_matrices_comparison.png`: Side-by-side confusion matrices
- `detailed_comparison.json`: Complete results in JSON format

## Performance Metrics

All models are evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training Time**: Total time and per-epoch
- **Model Size**: Number of parameters

## Troubleshooting

### Sensor Connection Issues
- Check USB port: `ls /dev/ttyUSB*`
- Verify baud rate matches sensor configuration
- Ensure proper permissions: `sudo chmod 666 /dev/ttyUSB0`

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use CPU: Model will automatically fall back if CUDA unavailable

### Data Loading Errors
- Ensure data files exist in specified directory
- Check .npz file format is correct
- Run `step explore` to verify data integrity

### Model Training Fails
- Check data shape matches model input
- Verify number of classes matches labels
- Ensure sufficient GPU memory

## Tips for Best Results

1. **Data Collection**:
   - Collect diverse contact patterns for each shape
   - Ensure consistent pressure and contact area
   - Collect at least 100 samples per class

2. **Training**:
   - Start with CNN for best general performance
   - Use learning rate scheduling for better convergence
   - Monitor validation accuracy for overfitting

3. **Model Selection**:
   - CNN: Best for most tactile classification tasks
   - ResNet: Better for complex shapes with similar patterns
   - LSTM/BiLSTM: Good for time-series tactile data
   - CNN-LSTM: Best for spatiotemporal patterns

## Citation

If you use this pipeline in your research, please cite:

```
@software{tactile_classification_pipeline,
  title={Tactile Sensor Shape Classification Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tactile_encoder}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- PyTorch for deep learning framework
- scikit-learn for evaluation metrics
- OpenCV for visualization
