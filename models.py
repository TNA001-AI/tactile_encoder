"""
Neural network models for tactile sensor shape classification
Includes: MLP, CNN, ResNet, and LSTM architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron for tactile data classification"""

    def __init__(self, input_shape=(16, 32), num_classes=5, hidden_dims=[512, 256, 128]):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
            hidden_dims: list of hidden layer dimensions
        """
        super(MLP, self).__init__()

        self.input_shape = input_shape
        self.input_dim = input_shape[0] * input_shape[1]

        layers = []
        prev_dim = self.input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)


class CNN(nn.Module):
    """Convolutional Neural Network for tactile data classification"""

    def __init__(self, input_shape=(16, 32), num_classes=5):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
        """
        super(CNN, self).__init__()

        self.input_shape = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Calculate the size after convolutions and pooling
        # After 3 pooling layers: H/8 x W/8
        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        self.feature_dim = 128 * feature_h * feature_w

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture for tactile data classification"""

    def __init__(self, input_shape=(16, 32), num_classes=5, num_blocks=[2, 2, 2]):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
            num_blocks: list of number of residual blocks per stage
        """
        super(ResNet, self).__init__()

        self.input_shape = input_shape
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)

        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DeepCNN(nn.Module):
    """Deeper CNN with more layers for better feature extraction"""

    def __init__(self, input_shape=(16, 32), num_classes=5):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
        """
        super(DeepCNN, self).__init__()

        self.input_shape = input_shape

        # Deeper convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout = nn.Dropout(0.5)

        # After 3 pooling layers: H/8 x W/8
        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        self.feature_dim = 256 * feature_h * feature_w

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Conv block 4 (no pooling)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class VGGLike(nn.Module):
    """VGG-style architecture adapted for tactile data"""

    def __init__(self, input_shape=(16, 32), num_classes=5):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
        """
        super(VGGLike, self).__init__()

        self.input_shape = input_shape

        # VGG-style blocks: conv-conv-pool
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)
        x = self.pool(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)
        x = self.pool(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.bn3(x)
        x = self.pool(x)

        # Global pooling and classifier
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AttentionCNN(nn.Module):
    """CNN with spatial attention mechanism for tactile data"""

    def __init__(self, input_shape=(16, 32), num_classes=5):
        """
        Args:
            input_shape: tuple (height, width) of tactile sensor
            num_classes: number of output classes
        """
        super(AttentionCNN, self).__init__()

        self.input_shape = input_shape

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Spatial attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Dual pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)

        # Concatenate both pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.fc(x)

        return x


def get_model(model_name, input_shape=(16, 32), num_classes=5, **kwargs):
    """
    Factory function to get model by name

    Args:
        model_name: str, one of ['mlp', 'cnn', 'resnet', 'deepcnn', 'vgg', 'attention']
        input_shape: tuple (height, width) of tactile sensor
        num_classes: number of output classes
        **kwargs: additional model-specific parameters

    Returns:
        model instance
    """
    models = {
        'mlp': MLP,
        'cnn': CNN,
        'resnet': ResNet,
        'deepcnn': DeepCNN,
        'vgg': VGGLike,
        'attention': AttentionCNN
    }

    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models.keys())}")

    return models[model_name.lower()](input_shape=input_shape, num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    # Test all models
    batch_size = 4
    input_shape = (16, 32)
    num_classes = 5

    x = torch.randn(batch_size, *input_shape)

    print("Testing all models:")
    print("=" * 60)

    model_names = ['mlp', 'cnn', 'resnet', 'deepcnn', 'vgg', 'attention']

    for model_name in model_names:
        model = get_model(model_name, input_shape=input_shape, num_classes=num_classes)
        output = model(x)

        num_params = sum(p.numel() for p in model.parameters())

        print(f"\n{model_name.upper()}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of parameters: {num_params:,}")
