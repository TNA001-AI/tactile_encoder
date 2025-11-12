"""
Online Evaluation Script for Tactile Sensor Shape Classification
Connects to tactile sensor and performs real-time shape classification using trained models
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Suppress Qt wayland warning

import numpy as np
import torch
import cv2
import time
import json
import argparse
import serial
import threading
from datetime import datetime
from models import get_model
from scipy.ndimage import gaussian_filter
from config_utils import load_config, override_config_from_args


def temporal_filter(new_frame, prev_frame, alpha=0.2):
    """
    Apply temporal smoothing filter (same as original script).
    'alpha' determines the blending factor.
    A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame


class TactileOnlineEvaluator:
    def __init__(self, model_path, model_name='cnn', config=None):
        """
        Initialize online evaluator
        
        Args:
            model_path: path to trained model (.pth file)
            model_name: type of model (cnn, resnet, etc.)
            config: ConfigManager instance
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.model_name = model_name
        self.model_path = model_path
        
        # Sensor configuration
        sensor_config = config.get_sensor_config()
        self.port = sensor_config['port']
        self.baud = sensor_config['baud_rate']
        self.sensor_shape = tuple(sensor_config['shape'])
        self.timeout = sensor_config.get('timeout', 1)
        
        # Data processing parameters
        data_config = config.get_data_config()
        self.THRESHOLD = data_config['threshold']
        self.NOISE_SCALE = data_config['noise_scale']
        self.gaussian_sigma = data_config['gaussian_sigma']
        self.temporal_alpha = data_config.get('temporal_alpha', 0.2)
        
        # Evaluation configuration
        eval_config = config.get_evaluation_config()
        self.min_confidence = eval_config.get('min_confidence', 0.5)
        self.smooth_predictions = eval_config.get('smooth_predictions', True)
        self.smoothing_window = eval_config.get('smoothing_window', 5)
        self.update_interval = eval_config.get('update_interval', 0.5)
        
        # Sensor data
        self.contact_data_norm = np.zeros(self.sensor_shape)
        self.flag = False
        self.median = None
        self.serDev = None
        self.serialThread = None
        self.prev_frame = np.zeros(self.sensor_shape)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.load_model()
        
        # Prediction tracking
        self.predictions = []
        self.prediction_history = []
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
    def load_model(self):
        """Load trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model info from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                self.class_names = config.get('class_names', ['unknown'])
                input_shape = config.get('input_shape', self.sensor_shape)
                num_classes = len(self.class_names)
            else:
                # Try to load from results.json in same directory
                results_path = os.path.join(os.path.dirname(self.model_path), 'results.json')
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    # Try both 'label_names' and 'class_names' for compatibility
                    self.class_names = results.get('label_names', results.get('class_names', ['unknown']))
                    input_shape = tuple(results.get('input_shape', self.sensor_shape))
                    num_classes = results.get('num_classes', len(self.class_names))
                else:
                    # Default fallback
                    self.class_names = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
                    input_shape = self.sensor_shape
                    num_classes = 5
            
            # Create model
            self.model = get_model(self.model_name, input_shape=input_shape, num_classes=num_classes)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully: {self.model_name}")
            print(f"✓ Classes: {self.class_names}")
            print(f"✓ Device: {self.device}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def readThread(self):
        """Thread for reading serial data from tactile sensor"""
        backup = None
        current = None
        
        while True:
            if self.serDev.in_waiting > 0:
                try:
                    line = self.serDev.readline().decode('utf-8').strip()
                except:
                    line = ""
                
                if len(line) < 10:
                    if current is not None and len(current) == self.sensor_shape[0]:
                        backup = np.array(current)
                        current = []
                        
                        if self.median is not None:
                            # Process data (EXACT same as original script)
                            contact_data = backup - self.median - self.THRESHOLD
                            contact_data = np.clip(contact_data, 0, 100)
                            
                            # Apply normalization (EXACT same as original script)
                            if np.max(contact_data) < self.THRESHOLD:
                                self.contact_data_norm = contact_data / self.NOISE_SCALE
                            else:
                                self.contact_data_norm = contact_data / np.max(contact_data)
                            
                            self.flag = True
                
                else:
                    try:
                        values = [int(x) for x in line.split()]  # Split on whitespace, not comma
                        if len(values) == self.sensor_shape[1]:
                            if current is None:
                                current = []
                            current.append(values)
                    except:
                        pass
    
    def start_sensor(self):
        """Initialize and start sensor connection"""
        try:
            print(f"🔌 Connecting to sensor on {self.port} at {self.baud} baud...")
            self.serDev = serial.Serial(self.port, self.baud)  # No timeout, same as original
            self.serDev.flush()  # Add flush like original
            time.sleep(1)
            
            print("🔌 Connected to sensor, calibrating...")
            
            # Collect baseline data for median calculation (EXACT same as original)
            data_tac = []
            num = 0
            current = None
            
            print("📊 Collecting 30 frames for calibration...")
            while True:
                if self.serDev.in_waiting > 0:
                    try:
                        line = self.serDev.readline().decode('utf-8').strip()
                    except:
                        line = ""
                    
                    if len(line) < 10:
                        if current is not None and len(current) == self.sensor_shape[0]:
                            backup = np.array(current)
                            data_tac.append(backup)
                            num += 1
                            if num > 30:  # Same as original: collect 30 frames
                                break
                        current = []
                        continue
                        
                    if current is not None:
                        str_values = line.split()  # Same as original
                        int_values = [int(val) for val in str_values]
                        matrix_row = int_values
                        current.append(matrix_row)
            
            # Calculate median (EXACT same as original)
            data_tac = np.array(data_tac)
            self.median = np.median(data_tac, axis=0)
            print("✓ Calibration complete - Finish Initialization!")
            
            # Start reading thread
            self.serialThread = threading.Thread(target=self.readThread, daemon=True)
            self.serialThread.start()
            
            print("✓ Sensor started successfully")
            
        except Exception as e:
            raise Exception(f"Failed to start sensor: {e}")
    
    def predict(self, tactile_data):
        """Make prediction on tactile data"""
        try:
            # Prepare input tensor
            input_tensor = torch.FloatTensor(tactile_data).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                return predicted_class, confidence_score, probabilities.cpu().numpy()[0]
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def run_evaluation(self, min_confidence=None, smooth_predictions=None):
        """Run online evaluation loop"""
        # Use config values if not provided
        if min_confidence is None:
            min_confidence = self.min_confidence
        if smooth_predictions is None:
            smooth_predictions = self.smooth_predictions
            
        print("\n" + "="*70)
        print("ONLINE TACTILE SHAPE CLASSIFICATION")
        print("="*70)
        print(f"Model: {self.model_name.upper()}")
        print(f"Classes: {', '.join(self.class_names)}")
        print(f"Min confidence: {min_confidence:.2f}")
        print(f"Smoothing: {smooth_predictions}")
        print("\nPress 'q' to quit, 's' to save prediction history")
        print("="*70)
        
        # Create display window (larger size for better visibility)
        WINDOW_WIDTH = self.sensor_shape[1] * 30   # 32 * 30 = 960
        WINDOW_HEIGHT = self.sensor_shape[0] * 30 + 100  # 16 * 30 + 100 = 580 (include text area)
        print("Creating display window...")
        cv2.namedWindow('Tactile Data', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tactile Data', WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2)  # Double the window size
        
        # Test display with a simple image (match window size)
        test_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        test_img[:] = (50, 50, 50)  # Dark gray
        cv2.putText(test_img, 'Waiting for sensor data...', (WINDOW_WIDTH//4, WINDOW_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Tactile Data', test_img)
        cv2.waitKey(1)  # Process the display
        print("Test window displayed")
        
        try:
            last_update = time.time()
            prediction_buffer = []
            buffer_size = self.smoothing_window if smooth_predictions else 1
            
            # Debug counters
            loop_count = 0
            data_received_count = 0
            
            # Initialize temporal filter (like original script)
            prev_frame = np.zeros_like(self.contact_data_norm)
            
            print("Starting evaluation loop...")
            print("Waiting for sensor data... (press Ctrl+C to abort)")
            
            while True:
                loop_count += 1
                
                # Debug: Print status every 1000 loops
                if loop_count % 1000 == 0:
                    print(f"Loop {loop_count}, Data received: {data_received_count}, Flag: {self.flag}")
                
                if self.flag:
                    data_received_count += 1
                    current_time = time.time()
                    
                    # Make prediction
                    predicted_class, confidence, probabilities = self.predict(self.contact_data_norm)
                    
                    if predicted_class is not None:
                        prediction_buffer.append((predicted_class, confidence))
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # Smooth predictions if enabled
                        if smooth_predictions and len(prediction_buffer) >= buffer_size:
                            # Use most common prediction with high enough confidence
                            high_conf_predictions = [p for p in prediction_buffer if p[1] >= min_confidence]
                            if high_conf_predictions:
                                from collections import Counter
                                most_common = Counter([p[0] for p in high_conf_predictions]).most_common(1)
                                if most_common:
                                    final_prediction = most_common[0][0]
                                    final_confidence = max([p[1] for p in high_conf_predictions if p[0] == final_prediction])
                                else:
                                    final_prediction = predicted_class
                                    final_confidence = confidence
                            else:
                                final_prediction = predicted_class
                                final_confidence = confidence
                        else:
                            final_prediction = predicted_class
                            final_confidence = confidence
                        
                        # Update tracking
                        self.last_prediction = final_prediction
                        self.prediction_confidence = final_confidence
                        
                        # Store prediction with timestamp
                        prediction_record = {
                            'timestamp': current_time,
                            'predicted_class': final_prediction,
                            'class_name': self.class_names[final_prediction],
                            'confidence': final_confidence,
                            'all_probabilities': probabilities.tolist() if probabilities is not None else None
                        }
                        self.prediction_history.append(prediction_record)
                    
                    # Apply temporal filter (like original script)
                    temp_filtered_data = temporal_filter(self.contact_data_norm, prev_frame, alpha=self.temporal_alpha)
                    prev_frame = temp_filtered_data
                    
                    # Scale to 0-255 and convert to uint8 (like original script)
                    temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)
                    
                    # Apply VIRIDIS colormap (like original script)
                    display_img = cv2.applyColorMap(temp_filtered_data_scaled, cv2.COLORMAP_VIRIDIS)

                    # Resize to proper window size (32*30 x 16*30 = 960 x 480)
                    WINDOW_WIDTH = self.sensor_shape[1] * 30  # 960
                    WINDOW_HEIGHT = self.sensor_shape[0] * 30  # 480
                    display_img_resized = cv2.resize(display_img, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)

                    # Create blank area below for prediction text
                    text_area_height = 100
                    text_area = np.zeros((text_area_height, WINDOW_WIDTH, 3), dtype=np.uint8)

                    # Add prediction text to blank area (only if confidence >= min_confidence)
                    if self.last_prediction is not None and self.prediction_confidence >= min_confidence:
                        class_name = self.class_names[self.last_prediction]
                        prediction_text = f"Prediction: {class_name}"
                        confidence_text = f"Confidence: {self.prediction_confidence:.3f}"

                        # Draw text on blank area
                        cv2.putText(text_area, prediction_text, (10, 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(text_area, confidence_text, (10, 75),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Combine tactile data (top) and prediction area (bottom)
                    combined_display = np.vstack([display_img_resized, text_area])

                    cv2.imshow('Tactile Data', combined_display)
                    self.flag = False
                    
                    # Console output (throttled)
                    if current_time - last_update > self.update_interval:  # Console update interval
                        if self.last_prediction is not None:
                            class_name = self.class_names[self.last_prediction]
                            if self.prediction_confidence >= min_confidence:
                                print(f"\033[92mPrediction: {class_name} (confidence: {self.prediction_confidence:.3f})\033[0m")
                            else:
                                # print(f"Low confidence: {class_name} ({self.prediction_confidence:.3f})")
                                continue
                                
                        last_update = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_prediction_history()
                    
        except KeyboardInterrupt:
            print("\nEvaluation stopped by user")
        finally:
            cv2.destroyAllWindows()
            self.close()
    
    def save_prediction_history(self):
        """Save prediction history to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_history_{timestamp}.json"
        
        summary = {
            'model_info': {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'class_names': self.class_names
            },
            'session_info': {
                'start_time': self.prediction_history[0]['timestamp'] if self.prediction_history else None,
                'end_time': self.prediction_history[-1]['timestamp'] if self.prediction_history else None,
                'total_predictions': len(self.prediction_history)
            },
            'predictions': self.prediction_history
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Prediction history saved to: {filename}")
    
    def close(self):
        """Close sensor connection"""
        try:
            if self.serDev:
                self.serDev.close()
                print("✓ Sensor connection closed")
        except:
            pass


def find_best_model(results_dir='./results'):
    """Find the best trained model automatically"""
    try:
        best_model = None
        best_accuracy = 0
        
        for model_dir in os.listdir(results_dir):
            model_path = os.path.join(results_dir, model_dir)
            if os.path.isdir(model_path):
                results_file = os.path.join(model_path, 'results.json')
                model_file = os.path.join(model_path, 'best_model.pth')
                
                if os.path.exists(results_file) and os.path.exists(model_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    accuracy = results.get('test_results', {}).get('accuracy', 0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        # Extract base model name (everything before first underscore or timestamp)
                        base_model_name = model_dir.split('_')[0]
                        best_model = {
                            'name': base_model_name,
                            'full_name': model_dir,
                            'path': model_file,
                            'accuracy': accuracy
                        }
        
        return best_model
    except:
        return None


def main():
    """Main function with config support"""
    parser = argparse.ArgumentParser(description='Online tactile shape classification evaluation')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--model-path', type=str, help='Path to model file (overrides auto-detection)')
    parser.add_argument('--model-name', type=str, help='Model architecture name')
    parser.add_argument('--port', type=str, help='Sensor port (overrides config)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold (overrides config)')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable prediction smoothing')
    
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
    
    print("🤖 Tactile Sensor Online Evaluation")
    print("="*50)
    
    # Check for trained models in both possible locations
    results_dir = config.get('paths.results_dir', './results')
    comparison_dir = config.get('paths.comparison_dir', './comparison_results')
    
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
        print("Please train a model first using:")
        print("  python train.py")
        print("  or")
        print("  python quick_start.py")
        return
    
    # Determine model to use
    if args.model_path and args.model_name:
        model_path = args.model_path
        model_name = args.model_name
    else:
        # Find best model automatically
        best_model = find_best_model(search_dir)
        
        if best_model:
            print(f"🎯 Auto-detected best model: {best_model['full_name']} (accuracy: {best_model['accuracy']:.3f})")
            model_name = best_model['name']  # Use base name for model loading
            model_path = best_model['path']
        else:
            print("❌ No valid trained models found")
            return
    
    # Get configuration values
    eval_config = config.get_evaluation_config()
    min_confidence = args.min_confidence or eval_config.get('min_confidence', 0.5)
    smooth_predictions = not args.no_smoothing and eval_config.get('smooth_predictions', True)
    
    print(f"\n🚀 Starting evaluation with {model_name}...")
    print(f"📊 Configuration:")
    print(f"  - Port: {config.get('sensor.port')}")
    print(f"  - Min confidence: {min_confidence}")
    print(f"  - Smoothing: {smooth_predictions}")
    
    try:
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
        

def manual_model_selection():
    """Manual model selection interface"""
    print("\nAvailable trained models:")
    models = []
    
    try:
        for i, model_dir in enumerate(os.listdir('./results')):
            model_path = os.path.join('./results', model_dir)
            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, 'best_model.pth')
                results_file = os.path.join(model_path, 'results.json')
                
                if os.path.exists(model_file):
                    accuracy = "unknown"
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            accuracy = f"{results.get('test_results', {}).get('accuracy', 0):.3f}"
                        except:
                            pass
                    
                    models.append((model_dir, model_file))
                    print(f"  {i+1}. {model_dir} (accuracy: {accuracy})")
        
        if not models:
            print("❌ No trained models found!")
            return None, None
        
        choice = input(f"\nSelect model (1-{len(models)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_name, model_path = models[idx]
                return model_name, model_path
            else:
                print("Invalid selection")
                return None, None
        except:
            print("Invalid input")
            return None, None
            
    except Exception as e:
        print(f"Error scanning models: {e}")
        return None, None


if __name__ == '__main__':
    main()