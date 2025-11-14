"""
Dataset and DataLoader utilities for tactile sensor data
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob


class TactileDataset(Dataset):
    """PyTorch Dataset for tactile sensor data"""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: numpy array of shape (N, H, W) where N is number of samples
            labels: numpy array of shape (N,) with string or integer labels
            transform: optional transform to apply to data
        """
        self.data = torch.FloatTensor(data)

        # Encode labels if they are strings
        if isinstance(labels[0], str):
            self.label_encoder = LabelEncoder()
            self.labels = torch.LongTensor(self.label_encoder.fit_transform(labels))
            self.label_names = self.label_encoder.classes_
        else:
            self.labels = torch.LongTensor(labels)
            self.label_encoder = None
            self.label_names = None

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def get_num_classes(self):
        """Return number of unique classes"""
        return len(torch.unique(self.labels))

    def get_label_names(self):
        """Return class names if available"""
        return self.label_names


class TactileDataLoader:
    """Utility class for loading and preparing tactile data"""

    def __init__(self, data_dir='./tactile_data', eval_data_dir=None, test_size=0.2, val_size=0.1, random_state=42):
        """
        Args:
            data_dir: directory containing training .npz files
            eval_data_dir: optional directory containing separate evaluation dataset
            test_size: fraction of data for testing (ignored if eval_data_dir is provided)
            val_size: fraction of training data for validation
            random_state: random seed for reproducibility
        """
        self.data_dir = data_dir
        self.eval_data_dir = eval_data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.label_encoder = None

    def load_data_from_directory(self):
        """Load all .npz files from directory and combine them"""
        npz_files = glob.glob(os.path.join(self.data_dir, '*.npz'))

        if not npz_files:
            raise ValueError(f"No .npz files found in {self.data_dir}")

        all_data = []
        all_labels = []

        print(f"Loading data from {len(npz_files)} files...")
        for npz_file in npz_files:
            data = np.load(npz_file)
            all_data.append(data['data'])
            all_labels.append(data['labels'])
            print(f"  Loaded {npz_file}: {data['data'].shape[0]} samples")

        # Combine all data
        data = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        print(f"\nTotal samples loaded: {data.shape[0]}")
        print(f"Data shape: {data.shape}")
        print(f"Unique labels: {np.unique(labels)}")

        return data, labels

    def load_single_file(self, filepath):
        """Load data from a single .npz file"""
        data = np.load(filepath)
        return data['data'], data['labels']

    def prepare_dataloaders(self, batch_size=32, num_workers=4, shuffle=True):
        """
        Load data and create train/val/test dataloaders

        Args:
            batch_size: batch size for dataloaders
            num_workers: number of worker processes for data loading
            shuffle: whether to shuffle training data

        Returns:
            train_loader, val_loader, test_loader, num_classes, label_names
        """
        # Load training data
        data, labels = self.load_data_from_directory()

        # If separate eval dataset is provided, use it for testing
        if self.eval_data_dir and os.path.exists(self.eval_data_dir):
            print(f"\nUsing separate evaluation dataset from: {self.eval_data_dir}")

            # Load eval data
            eval_files = glob.glob(os.path.join(self.eval_data_dir, '*.npz'))
            if not eval_files:
                print(f"Warning: No .npz files found in {self.eval_data_dir}, falling back to split")
                # Fall back to splitting from training data
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    data, labels, test_size=self.test_size,
                    random_state=self.random_state, stratify=labels
                )
            else:
                all_eval_data = []
                all_eval_labels = []
                print(f"Loading evaluation data from {len(eval_files)} files...")
                for npz_file in eval_files:
                    eval_npz = np.load(npz_file)
                    all_eval_data.append(eval_npz['data'])
                    all_eval_labels.append(eval_npz['labels'])
                    print(f"  Loaded {npz_file}: {eval_npz['data'].shape[0]} samples")

                X_test = np.concatenate(all_eval_data, axis=0)
                y_test = np.concatenate(all_eval_labels, axis=0)
                X_trainval = data
                y_trainval = labels
                print(f"Total evaluation samples: {len(X_test)}")
        else:
            # Split into train+val and test
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                data, labels, test_size=self.test_size,
                random_state=self.random_state, stratify=labels
            )

        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=self.val_size,
            random_state=self.random_state, stratify=y_trainval
        )

        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")

        # Create datasets
        train_dataset = TactileDataset(X_train, y_train)
        val_dataset = TactileDataset(X_val, y_val)
        test_dataset = TactileDataset(X_test, y_test)

        # Get label information
        num_classes = train_dataset.get_num_classes()
        label_names = train_dataset.get_label_names()

        # Check if CUDA is available for pin_memory
        import torch
        use_pin_memory = torch.cuda.is_available()

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, pin_memory=use_pin_memory
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
        )

        return train_loader, val_loader, test_loader, num_classes, label_names

    def get_data_statistics(self):
        """Calculate and return statistics about the data"""
        data, labels = self.load_data_from_directory()

        unique_labels, counts = np.unique(labels, return_counts=True)

        stats = {
            'total_samples': len(data),
            'data_shape': data.shape,
            'num_classes': len(unique_labels),
            'class_names': unique_labels.tolist(),
            'class_counts': dict(zip(unique_labels, counts)),
            'data_min': float(np.min(data)),
            'data_max': float(np.max(data)),
            'data_mean': float(np.mean(data)),
            'data_std': float(np.std(data))
        }

        return stats


# Data augmentation transforms
class RandomNoise:
    """Add random Gaussian noise to data"""
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class RandomFlip:
    """Randomly flip the tactile image horizontally or vertically"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            # Horizontal flip
            x = torch.flip(x, dims=[1])
        if torch.rand(1) < self.p:
            # Vertical flip
            x = torch.flip(x, dims=[0])
        return x


class Normalize:
    """Normalize data to zero mean and unit variance"""
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.mean is None:
            self.mean = x.mean()
        if self.std is None:
            self.std = x.std()
        return (x - self.mean) / (self.std + 1e-8)


if __name__ == '__main__':
    # Example usage
    loader = TactileDataLoader(data_dir='./tactile_data')

    # Get statistics
    print("Data Statistics:")
    stats = loader.get_data_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Prepare dataloaders
    print("\nPreparing dataloaders...")
    train_loader, val_loader, test_loader, num_classes, label_names = \
        loader.prepare_dataloaders(batch_size=32)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {label_names}")

    # Test loading a batch
    for data, labels in train_loader:
        print(f"\nBatch shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        break
