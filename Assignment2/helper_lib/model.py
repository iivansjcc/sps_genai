import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# 1) Fully Connected Neural Net (Module 4 FCNN)
#    Flatten → Linear → ReLU → Dropout → Linear
# =====================================================
class FCNN(nn.Module):
    def __init__(self,
                 in_features: int = 3 * 32 * 32,  # adjust if input size differs
                 hidden: int = 256,
                 num_classes: int = 10,
                 dropout_p: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =====================================================
# 2) Simple CNN (Baseline)
#    Conv → ReLU → Pool × 2 → FC → FC
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # for 32×32 input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =====================================================
# 3) Enhanced CNN (Module 4 Convolution Practical)
#    Conv → BN → ReLU → Pool × 4 + FC + Dropout
#    Channels: 3→16→32→64→128 ; FC1 input = 128×2×2
# =====================================================
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # For 32×32 inputs → after 4 pools: 2×2 spatial size
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

        # For 64×64 inputs
        self.gap = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =====================================================
# 4) Model Selector
# =====================================================
def get_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Choose a model by name:
      - 'FCNN'         : Fully-connected network (Module 4 FCNN)
      - 'CNN'          : Simple CNN baseline
      - 'EnhancedCNN'  : CNN with 4 Conv blocks, BN, Dropout
    Extra kwargs (e.g., in_features, hidden, dropout_p) are passed through.
    """
    name = model_name.strip().lower()
    if name == "fcnn":
        return FCNN(num_classes=num_classes, **kwargs)
    elif name == "cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif name == "enhancedcnn":
        return EnhancedCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from 'FCNN', 'CNN', 'EnhancedCNN'.")