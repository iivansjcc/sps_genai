# Assignment4/helper_lib/data.py

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def to_neg_one_one(x):
    """Convert [0,1] tensor → [-1,1]"""
    return x * 2.0 - 1.0


def _get_data_root() -> Path:
    """Return Assignment4/data/cifar10 independent of working directory."""
    here = Path(__file__).resolve()
    assignment_dir = here.parents[1]  # .../Assignment4
    return assignment_dir / "data" / "cifar10"


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:

    root = _get_data_root()

    train_tfm_list = []
    if augment:
        train_tfm_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ]

    train_tfm_list += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    train_transform = transforms.Compose(train_tfm_list)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=train_transform
    )

    test_ds = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def get_cifar10_diffusion_loader(
    batch_size: int = 128,
    num_workers: int = 0,   # IMPORTANT: avoid multiprocessing pickling issues
):

    root = _get_data_root()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_neg_one_one),
    ])

    train_ds = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader