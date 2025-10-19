from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Union

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate the model on a given dataset.

    Returns:
        avg_loss (float): Average loss over dataset
        accuracy (float): Accuracy between 0 and 1
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy