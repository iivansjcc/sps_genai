import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib import (
    get_loaders, get_model, train_model, evaluate_model,
    save_model, print_model_summary,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train/Evaluate CNN or FCNN on ImageFolder data")
    # data
    p.add_argument("--train_dir", type=str, default="data/train", help="Path to training data")
    p.add_argument("--test_dir",  type=str, default="data/test",  help="Path to test/val data")
    p.add_argument("--img_size",  type=int, default=32, help="Resize images to this square size")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    # model
    p.add_argument("--model", type=str, default="enhancedcnn",
                   choices=["fcnn", "cnn", "enhancedcnn"])
    p.add_argument("--num_classes", type=int, default=10)
    # train
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping (None to disable)")
    # save
    p.add_argument("--save_path", type=str, default="checkpoints/model.pth")
    p.add_argument("--save_with_timestamp", action="store_true")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--eval_during_train", action="store_true",
                   help="Use test_dir as val_loader to report val loss/acc during training")
    return p.parse_args()

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) Data
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std  = (0.229, 0.224, 0.225)
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    train_ds = datasets.CIFAR10(root="data/cifar10", train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

    # --- Auto-detect number of classes from the training dataset ---
    if hasattr(train_ds, "classes"):
        args.num_classes = len(train_ds.classes)
        print(f"Detected {args.num_classes} classes: {train_ds.classes}")
    else:
        print(f"Warning: could not detect classes; using args.num_classes={args.num_classes}")

    # 2) Model
    model = get_model(args.model, num_classes=args.num_classes)
    print_model_summary(model)

    # 3) Optim/criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) Train
    val_loader = test_loader if args.eval_during_train else None
    model, history = train_model(
        model, train_loader, criterion, optimizer,
        device=args.device, epochs=args.epochs,
        val_loader=val_loader, max_grad_norm=args.max_grad_norm,
        log_every=args.log_every, return_history=True
    )

    # 5) Evaluate (final on test set)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device=args.device)
    print(f"\nFinal Test → loss: {test_loss:.4f} | acc: {test_acc:.2%}")

    # 6) Save checkpoint
    saved = save_model(model, args.save_path, with_timestamp=args.save_with_timestamp)
    print(f"Model saved to: {saved}")

if __name__ == "__main__":
    main()
