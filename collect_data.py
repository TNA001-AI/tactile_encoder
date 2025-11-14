"""
Tactile Sensor Data Collection Script for Shape Classification
Collects labeled tactile sensor data for training classification models
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Suppress Qt wayland warning

import numpy as np
import serial
import threading
import cv2
import time
import json
import argparse
from datetime import datetime
from scipy.ndimage import gaussian_filter
from config_utils import load_config, override_config_from_args

class TactileDataCollector:
    def __init__(self, config=None):
        """
        Initialize tactile data collector
        
        Args:
            config: ConfigManager instance or None to load default
        """
        if config is None:
            from config_utils import load_config
            config = load_config()
        
        self.config = config
        
        # Sensor configuration
        sensor_config = config.get_sensor_config()
        self.port = sensor_config['port']
        self.baud = sensor_config['baud_rate']
        self.sensor_shape = tuple(sensor_config['shape'])
        self.timeout = sensor_config.get('timeout', 1)
        
        # Data collection configuration
        data_config = config.get_data_config()
        self.THRESHOLD = data_config['threshold']
        self.NOISE_SCALE = data_config['noise_scale']
        self.gaussian_sigma = data_config['gaussian_sigma']
        self.shape_labels = data_config['shape_labels']
        self.samples_per_shape = data_config['samples_per_shape']
        self.data_dir = data_config['data_dir']
        
        # Initialize sensor data
        self.contact_data_norm = np.zeros(self.sensor_shape)
        self.flag = False
        self.median = None
        self.serDev = None
        self.serialThread = None
        self.prev_frame = np.zeros(self.sensor_shape)
        self.running = False  # Flag to control thread execution

        # Collection tracking
        self.collected_samples = []
        self.current_label = None

    def readThread(self):
        """Thread for reading serial data from tactile sensor"""
        backup = None
        current = None

        # Main reading loop
        while self.running:
            if self.serDev and self.serDev.is_open and self.serDev.in_waiting > 0:
                try:
                    line = self.serDev.readline().decode('utf-8').strip()
                except:
                    line = ""

                if len(line) < 10:
                    if current is not None and len(current) == self.sensor_shape[0]:
                        backup = np.array(current)
                    current = []

                    if backup is not None:
                        contact_data = backup - self.median - self.THRESHOLD
                        contact_data = np.clip(contact_data, 0, 100)

                        if np.max(contact_data) < self.THRESHOLD:
                            self.contact_data_norm = contact_data / self.NOISE_SCALE
                        else:
                            self.contact_data_norm = contact_data / np.max(contact_data)

                        # Apply gaussian smoothing
                        self.contact_data_norm = gaussian_filter(self.contact_data_norm, sigma=self.gaussian_sigma)
                    continue

                if current is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    current.append(int_values)
            else:
                # Small sleep to prevent busy-waiting
                time.sleep(0.001)

    def temporal_filter(self, new_frame, alpha=0.2):
        """Apply temporal smoothing filter"""
        filtered = alpha * new_frame + (1 - alpha) * self.prev_frame
        self.prev_frame = filtered
        return filtered

    def start_sensor(self):
        """Initialize serial connection and start reading thread"""
        print(f"Connecting to sensor on {self.port} at {self.baud} baud...")
        self.serDev = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self.serDev.flush()

        # Calibration phase - collect 30 frames for baseline median
        print("Calibrating sensor... Collecting baseline data...")
        data_tac = []
        num = 0
        current = None

        while num <= 30:
            if self.serDev.in_waiting > 0:
                try:
                    line = self.serDev.readline().decode('utf-8').strip()
                except:
                    line = ""

                if len(line) < 10:
                    if current is not None and len(current) == self.sensor_shape[0]:
                        data_tac.append(np.array(current))
                        num += 1
                    current = []
                    continue

                if current is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    current.append(int_values)

        # Calculate baseline median
        self.median = np.median(data_tac, axis=0)
        print("Sensor calibration complete!")

        # Start continuous reading thread
        self.flag = True
        self.running = True
        self.serialThread = threading.Thread(target=self.readThread)
        self.serialThread.daemon = True
        self.serialThread.start()
        print("Sensor ready for data collection!")

    def visualize_data(self, data):
        """Visualize tactile data with color map"""
        # Apply temporal filter
        filtered_data = self.temporal_filter(data)

        # Scale to 0-255
        scaled_data = (filtered_data * 255).astype(np.uint8)

        # Apply color map
        colormap = cv2.applyColorMap(scaled_data, cv2.COLORMAP_VIRIDIS)

        return colormap

    def collect_samples(self, label, num_samples=100, save_dir='./tactile_data'):
        """
        Collect tactile sensor samples for a specific shape label

        Args:
            label: str, name of the shape (e.g., 'sphere', 'cube', 'cylinder')
            num_samples: int, number of samples to collect
            save_dir: str, directory to save collected data
        """
        self.current_label = label
        samples = []

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Collecting {num_samples} samples for label: {label}")
        print(f"Press 's' to save a sample, 'q' to finish early")
        print(f"{'='*60}\n")

        cv2.namedWindow("Tactile Data Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tactile Data Collection",
                        self.sensor_shape[1]*30, self.sensor_shape[0]*30)

        sample_count = 0

        while sample_count < num_samples:
            if self.flag:
                # Visualize current tactile data
                colormap = self.visualize_data(self.contact_data_norm)

                # Add text overlay
                info_text = f"Label: {label} | Samples: {sample_count}/{num_samples}"
                cv2.putText(colormap, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(colormap, "Press 's' to save, 'q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow("Tactile Data Collection", colormap)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    # Save current sample
                    sample = {
                        'data': self.contact_data_norm.copy(),
                        'label': label,
                        'timestamp': time.time()
                    }
                    samples.append(sample)
                    sample_count += 1
                    print(f"Sample {sample_count}/{num_samples} saved")
                    time.sleep(0.2)  # Prevent multiple saves

                elif key == ord('q'):
                    print("Collection stopped by user")
                    break

            time.sleep(0.01)

        cv2.destroyAllWindows()

        # Save samples to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'{label}_{timestamp}.npz')

        data_array = np.array([s['data'] for s in samples])
        labels_array = np.array([s['label'] for s in samples])

        np.savez(filename, data=data_array, labels=labels_array)
        print(f"\nSaved {len(samples)} samples to {filename}")

        return samples

    def collect_dataset(self, shape_labels, samples_per_shape=100, save_dir='./tactile_data'):
        """
        Collect a complete dataset with multiple shape labels

        Args:
            shape_labels: list of str, shape names to collect
            samples_per_shape: int, number of samples per shape
            save_dir: str, directory to save collected data
        """
        all_samples = []

        for label in shape_labels:
            print(f"\n\nPrepare to collect data for: {label}")
            input("Press Enter when ready...")

            samples = self.collect_samples(label, samples_per_shape, save_dir)
            all_samples.extend(samples)

            print(f"Completed collection for {label}")
            time.sleep(1)

        # Save combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join(save_dir, f'dataset_{timestamp}.npz')

        data_array = np.array([s['data'] for s in all_samples])
        labels_array = np.array([s['label'] for s in all_samples])

        np.savez(combined_file, data=data_array, labels=labels_array)

        # Save metadata
        metadata = {
            'shape_labels': shape_labels,
            'samples_per_shape': samples_per_shape,
            'total_samples': len(all_samples),
            'sensor_shape': self.sensor_shape,
            'timestamp': timestamp
        }

        metadata_file = os.path.join(save_dir, f'metadata_{timestamp}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"\n{'='*60}")
        print(f"Dataset collection complete!")
        print(f"Total samples: {len(all_samples)}")
        print(f"Combined dataset saved to: {combined_file}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"{'='*60}")

    def close(self):
        """Close serial connection"""
        # Stop the reading thread first
        self.running = False

        # Wait for thread to finish (with timeout)
        if self.serialThread and self.serialThread.is_alive():
            self.serialThread.join(timeout=1.0)

        # Close serial device
        if self.serDev and self.serDev.is_open:
            self.serDev.close()


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Collect tactile sensor data for shape classification')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--port', type=str, help='Sensor port (overrides config)')
    parser.add_argument('--shapes', type=str, nargs='+', help='Shape labels to collect')
    parser.add_argument('--samples', type=int, help='Samples per shape (overrides config)')
    parser.add_argument('--data-dir', type=str, help='Output directory (overrides config)')
    parser.add_argument('--eval', action='store_true', help='Collect evaluation dataset (saved to eval_data directory)')

    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("Using default configuration...")
        from config_utils import get_default_config
        config = get_default_config()
    
    # Override with command line arguments
    config = override_config_from_args(config, args)
    
    # Handle shapes argument
    if args.shapes:
        config.set('data_collection.shape_labels', args.shapes)

    # Handle evaluation dataset flag
    if args.eval:
        eval_dir = config.get('paths.eval_data_dir', './eval_data')
        if args.data_dir is None:
            config.set('data_collection.data_dir', eval_dir)
        dataset_type = "EVALUATION"
    else:
        dataset_type = "TRAINING"

    # Create collector
    collector = TactileDataCollector(config)

    print("\n" + "="*70)
    print(f"TACTILE {dataset_type} DATA COLLECTION")
    print("="*70)
    print(f"Sensor: {collector.port} @ {collector.baud} baud")
    print(f"Shapes: {collector.shape_labels}")
    print(f"Samples per shape: {collector.samples_per_shape}")
    print(f"Output directory: {collector.data_dir}")
    print("="*70)

    try:
        # Start sensor
        collector.start_sensor()

        # Collect dataset
        collector.collect_dataset(
            shape_labels=collector.shape_labels,
            samples_per_shape=collector.samples_per_shape,
            save_dir=collector.data_dir
        )

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        collector.close()


if __name__ == '__main__':
    main()
