import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _build_transforms(img_size: int = 32, train: bool = True):
    """
    Compose transforms for 3-channel images.
    Converts grayscale to RGB automatically and normalizes.
    """
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std  = (0.229, 0.224, 0.225)

    if train:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # If input can be grayscale, ensure 3 channels
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])
    return tfm


def get_data_loader(
    data_dir: str,
    batch_size: int = 32,
    train: bool = True,
    img_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
):
    """
    Build a single DataLoader from an ImageFolder at `data_dir`.

    Folder layout expected:
        data_dir/
          class_a/  img1.png, img2.png, ...
          class_b/  ...
    """
    if persistent_workers is None:
        # persistent_workers only valid when num_workers > 0
        persistent_workers = num_workers > 0

    tfm = _build_transforms(img_size=img_size, train=bool(train))
    dataset = datasets.ImageFolder(root=data_dir, transform=tfm)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return loader


def get_loaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 64,
    img_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """
    Convenience wrapper: returns (train_loader, test_loader).
    """
    train_loader = get_data_loader(
        train_dir, batch_size=batch_size, train=True,
        img_size=img_size, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = get_data_loader(
        test_dir, batch_size=batch_size, train=False,
        img_size=img_size, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader

