import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib import get_model
from helper_lib.data_loader import get_cifar10_loaders


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser("Train EnhancedCNN on CIFAR-10")
    p.add_argument("--model", type=str, default="enhancedcnn",
                   choices=["enhancedcnn", "simplecnn", "fcnn"])
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=pick_device())
    p.add_argument("--ckpt", type=str, default="checkpoints/model.pth")
    return p.parse_args()


def accuracy(logits, targets):
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1) Load CIFAR-10 explicitly
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    # 2) Auto-detect number of classes
    class_names = getattr(train_loader.dataset, "classes", [])
    num_classes = len(class_names) if class_names else 10
    print(f"Detected {num_classes} classes: {class_names}")

    # 3) Build model
    model = get_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) Train loop (simple & self-contained)
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(logits.detach(), y)
            n_batches += 1

        print(f"Epoch {epoch:03d}: "
              f"train_loss={running_loss/n_batches:.4f} "
              f"train_acc={running_acc/n_batches:.4f}")

        # quick eval each epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, n = 0.0, 0.0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                val_acc += accuracy(logits, y)
                n += 1
        print(f"           val_loss={val_loss/n:.4f} val_acc={val_acc/n:.4f}")
        model.train()

    # 5) Save checkpoint (what the API loads)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    print(f"Saved model to {args.ckpt}")


if __name__ == "__main__":
    main()