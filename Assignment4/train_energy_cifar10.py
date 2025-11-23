

from pathlib import Path

import torch

from helper_lib.data import get_cifar10_loaders
from helper_lib.energy_model import EnergyCNN
from helper_lib.trainer import train_energy


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps as mps
        if mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    device = pick_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_cifar10_loaders(batch_size=128, num_workers=2, augment=True)

    model = EnergyCNN(num_classes=10)
    here = Path(__file__).resolve()
    assignment_dir = here.parents[0]   # Assignment4
    ckpt_path = assignment_dir / "outputs" / "energy_cifar10.pth"

    train_energy(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=10,      # adjust for time/quality tradeoff
        lr=1e-3,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()