

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyCNN(nn.Module):
    """
    Simple CIFAR-10 CNN classifier.
    We treat the negative log-sum-exp of logits as the energy.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def compute_energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Example energy: E(x) = -logsumexp(logits).
    Lower energy means higher confidence.
    """
    # logsumexp over classes
    lse = torch.logsumexp(logits, dim=1)
    return -lse