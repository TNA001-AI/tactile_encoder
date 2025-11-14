# Tactile Sensor Shape Classification Pipeline

A complete machine learning pipeline for collecting tactile sensor data and training multiple deep learning models for shape classification tasks.

## Overview

This pipeline provides an end-to-end solution for:
1. **Data Collection**: Collect labeled tactile sensor data from a 16x32 tactile array
2. **Data Processing**: Load, preprocess, and split data for training
3. **Model Training**: Train multiple neural network architectures
4. **Model Comparison**: Compare and evaluate different models
5. **Visualization**: Generate comprehensive plots and reports


## Installation

1. Clone this repository or navigate to the project directory

2. Install dependencies:
```bash
conda create -n tactile python==3.12
conda activate tactile
pip install -r requirements.txt
```

## Quick Start

### Change the Port Permission

```bash
sudo chmod 666 /dev/ttyUSB0
```

### Run Quick Start

```bash
cd ~/tactile_encoder
python quick_start.py
```

The quick start menu now includes:
1. Collect training data
2. Collect evaluation data (new!)
3. Explore existing data
4. Train a single model (with wandb option)
5. Train and compare all models
6. Evaluate model online
7. Exit

### Step-by-Step Execution

#### Step 1: Collect Training Data
```bash
# Collect training data for different shapes
python collect_data.py
```

The data collector will:
- Connect to your tactile sensor (default: /dev/ttyUSB0)
- Initialize and calibrate the sensor
- Guide you through collecting samples for each shape
- Press 's' to save each sample, 'q' to finish early
- Save data as .npz files in `./tactile_data/`

#### Step 2: Collect Evaluation Data (Optional but Recommended)
```bash
# Collect a separate evaluation dataset
python collect_data.py --eval
```

This will save evaluation data to `./eval_data/`. Having a separate evaluation dataset helps:
- Better assess model generalization
- Avoid data leakage
- Get more reliable performance metrics

#### Step 3: Train Models
```bash
# Basic training
python train.py

# Training with wandb logging
python train.py --wandb --wandb-project my-tactile-project

# Training with evaluation dataset
python train.py --eval-data-dir ./eval_data

# Training with all options
python train.py --model cnn --wandb --eval-data-dir ./eval_data --epochs 100
```

**Wandb Integration:**
- Enable with `--wandb` flag
- Track training metrics in real-time
- Visualize loss, accuracy, and confusion matrices
- Compare multiple runs

#### Step 4: Compare Models
```bash
python compare_models.py
```

#### Step 5: Evaluate Online (Real-time Predictions)
```bash
python eval_online.py
```

Or use the quick start menu option 5.

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

### Example 3: Online Real-time Evaluation
```python
from eval_online import TactileOnlineEvaluator

# Create evaluator with trained model
evaluator = TactileOnlineEvaluator(
    model_path='./results/cnn/best_model.pth',
    model_name='cnn',
    port='/dev/ttyUSB0'
)

# Start sensor and run evaluation
evaluator.start_sensor()
evaluator.run_evaluation(min_confidence=0.5, smooth_predictions=True)
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
├── eval_online.py           # Real-time online evaluation
├── quick_start.py           # Interactive menu interface
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

The [config.json](config.json) file contains all configuration settings:

```json
{
  "sensor": {
    "port": "/dev/ttyUSB0",
    "baud_rate": 2000000,
    "shape": [16, 32]
  },
  "data_collection": {
    "shape_labels": ["sphere", "cube", "cylinder", "cone", "pyramid"],
    "samples_per_shape": 100,
    "data_dir": "./tactile_data"
  },
  "training": {
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 10
  },
  "paths": {
    "data_dir": "./tactile_data",
    "eval_data_dir": "./eval_data",
    "results_dir": "./results"
  },
  "wandb": {
    "enabled": false,
    "project": "tactile-classification",
    "entity": null,
    "tags": ["tactile", "shape-classification"]
  }
}
```

### Wandb Configuration

To enable wandb logging by default, edit [config.json](config.json):
```json
{
  "wandb": {
    "enabled": true,
    "project": "your-project-name",
    "entity": "your-wandb-username"
  }
}
```

Or use command line flags:
```bash
python train.py --wandb --wandb-project my-project
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

## Online Real-time Evaluation

The `eval_online.py` script provides real-time shape classification:

### Features
- **Live sensor feed**: Real-time tactile data visualization
- **Automatic model selection**: Uses best trained model by default
- **Confidence filtering**: Only show predictions above threshold
- **Temporal smoothing**: Average predictions over time for stability
- **Interactive controls**: 
  - Press 'q' to quit
  - Press 's' to save prediction history
- **Prediction history**: JSON logs with timestamps and confidence scores

### Usage
```bash
# Auto-detect best model and run
python eval_online.py

# Or use quick start menu
python quick_start.py  # Select option 5
```

### Configuration Options
- **Port**: Sensor serial port (default: `/dev/ttyUSB0`)
- **Min confidence**: Threshold for showing predictions (default: 0.5)
- **Smoothing**: Enable temporal prediction smoothing (default: yes)

### Output
- Real-time color-coded tactile visualization
- Live prediction display with confidence scores
- Prediction history saved as JSON files
- Console logging of high-confidence predictions

## Performance Metrics

All models are evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training Time**: Total time and per-epoch
- **Model Size**: Number of parameters

## Features

### Separate Evaluation Dataset
This pipeline supports collecting and using a separate evaluation dataset:
- **Collect**: `python collect_data.py --eval`
- **Train with eval data**: `python train.py --eval-data-dir ./eval_data`
- **Benefits**: Better generalization assessment, no data leakage, more reliable metrics

### Wandb Integration
Track your experiments with Weights & Biases:
- **Real-time metrics**: Monitor training loss, accuracy, and validation performance
- **Visualizations**: Automatic confusion matrix and training curve logging
- **Experiment tracking**: Compare multiple runs and hyperparameters
- **Setup**: Run `wandb login` first, then use `--wandb` flag

**Logged Metrics:**
- Training and validation loss/accuracy per epoch
- Test accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Learning rate schedule
- Model parameters and configuration

## Troubleshooting

### Sensor Connection Issues
- Check USB port: `ls /dev/ttyUSB*`
- Verify baud rate matches sensor configuration
- Ensure proper permissions: `sudo chmod 666 /dev/ttyUSB0`

### Wandb Issues
- **Not logged in**: Run `wandb login` and enter your API key
- **Offline mode**: Add `wandb offline` before training
- **Disable wandb**: Remove `--wandb` flag or set `"enabled": false` in config

## Credits

The hardware and base code are from Binghao Huang: https://docs.google.com/document/d/1XGyn-iV_wzRmcMIsyS3kwcrjxbnvblZAyigwbzDsX-E/edit?tab=t.0#heading=h.ny8zu0pq9mxy


