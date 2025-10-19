from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union

def _to_device(batch, device):
    """Move a (images, labels) batch to device."""
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device] = "cpu",
    max_grad_norm: Optional[float] = None,
) -> float:
    """
    Train for a single epoch; returns average loss.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for images, labels in data_loader:
        images, labels = _to_device((images, labels), device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)

def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device] = "cpu",
    epochs: int = 10,
    val_loader: Optional[DataLoader] = None,
    max_grad_norm: Optional[float] = None,
    log_every: int = 1,
    return_history: bool = False,
):
    """
    Generic supervised training loop.

    Args:
        model: torch model (e.g., FCNN/CNN/EnhancedCNN)
        data_loader: training dataloader
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        optimizer: optimizer (e.g., optim.Adam(...))
        device: 'cpu' or 'cuda'
        epochs: number of epochs
        val_loader: optional validation loader to report val loss/acc
        max_grad_norm: optional grad clipping value
        log_every: print every N epochs
        return_history: if True, also return a dict with per-epoch metrics

    Returns:
        model  (and optionally history: Dict[str, List[float]])
    """
    device = torch.device(device)
    model.to(device)

    history: Dict[str, List[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_acc"] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, data_loader, criterion, optimizer,
            device=device, max_grad_norm=max_grad_norm
        )
        history["train_loss"].append(train_loss)

        log_str = f"[Epoch {epoch:03d}/{epochs:03d}] train_loss={train_loss:.4f}"

        if val_loader is not None:
            with torch.no_grad():
                model.eval()
                total_loss, total_correct, total_samples = 0.0, 0, 0
                for images, labels in val_loader:
                    images, labels = _to_device((images, labels), device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += images.size(0)
                val_loss = total_loss / max(total_samples, 1)
                val_acc = total_correct / max(total_samples, 1)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                log_str += f" | val_loss={val_loss:.4f} val_acc={val_acc:.2%}"

        if (epoch % max(1, log_every)) == 0:
            print(log_str)

    return (model, history) if return_history else model