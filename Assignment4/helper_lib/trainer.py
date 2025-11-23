from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .diffusion_model import q_sample


def train_energy(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    ckpt_path: Optional[Path] = None,
) -> nn.Module:
    """
    Train the energy model (classifier) with cross-entropy on CIFAR-10.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[Energy] Epoch {ep:03d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved energy model to {ckpt_path}")

    return model


def train_diffusion(
    model: nn.Module,
    train_loader: DataLoader,
    schedule: Dict[str, torch.Tensor],
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 2e-4,
    ckpt_path: Optional[Path] = None,
) -> nn.Module:
    """
    Train diffusion model with simple DDPM objective:
      E[ || eps - eps_theta(x_t, t) ||^2 ]
    where x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"]
    T = schedule["T"]

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, _ in train_loader:
            x = x.to(device)  # (N,3,32,32), in [-1,1]
            b = x.size(0)

            t = torch.randint(0, T, (b,), device=device).long()
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            optimizer.zero_grad()
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            total += b

        avg_loss = total_loss / total
        print(f"[Diffusion] Epoch {ep:03d} | loss={avg_loss:.6f}")

    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved diffusion model to {ckpt_path}")

    return model