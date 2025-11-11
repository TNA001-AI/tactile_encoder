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
from datetime import datetime
from scipy.ndimage import gaussian_filter

class TactileDataCollector:
    def __init__(self, port='/dev/ttyUSB0', baud=2000000, sensor_shape=(16, 32)):
        self.port = port
        self.baud = baud
        self.sensor_shape = sensor_shape
        self.contact_data_norm = np.zeros(sensor_shape)
        self.flag = False
        self.median = None
        self.serDev = None
        self.serialThread = None
        self.prev_frame = np.zeros(sensor_shape)

        # Data collection parameters
        self.THRESHOLD = 30
        self.NOISE_SCALE = 50
        self.collected_samples = []
        self.current_label = None

    def readThread(self):
        """Thread for reading serial data from tactile sensor"""
        backup = None
        current = None

        # Main reading loop
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

                    if backup is not None:
                        contact_data = backup - self.median - self.THRESHOLD
                        contact_data = np.clip(contact_data, 0, 100)

                        if np.max(contact_data) < self.THRESHOLD:
                            self.contact_data_norm = contact_data / self.NOISE_SCALE
                        else:
                            self.contact_data_norm = contact_data / np.max(contact_data)
                    continue

                if current is not None:
                    str_values = line.split()
                    int_values = [int(val) for val in str_values]
                    current.append(int_values)

    def temporal_filter(self, new_frame, alpha=0.2):
        """Apply temporal smoothing filter"""
        filtered = alpha * new_frame + (1 - alpha) * self.prev_frame
        self.prev_frame = filtered
        return filtered

    def start_sensor(self):
        """Initialize serial connection and start reading thread"""
        print(f"Connecting to sensor on {self.port}...")
        self.serDev = serial.Serial(self.port, self.baud)
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
        if self.serDev:
            self.serDev.close()


if __name__ == '__main__':
    # Example usage
    collector = TactileDataCollector(port='/dev/ttyUSB1', baud=2000000)

    try:
        # Start sensor
        collector.start_sensor()

        # Define shapes to collect
        shapes = ['cube', 'cylinder', 'cone', 'pyramid']

        # Collect dataset
        collector.collect_dataset(
            shape_labels=shapes,
            samples_per_shape=100,
            save_dir='./tactile_data'
        )

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    finally:
        collector.close()
