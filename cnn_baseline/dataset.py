"""
dataset.py — Fashion-MNIST loader that reads from the YOLO-format directory
(subfolders named 0–9 for each class), reusing the already-downloaded dataset.

Mirrors YOLO's preprocessing: grayscale → 3-channel RGB, resize to 28×28,
normalize with ImageNet stats (same as YOLO classification pipeline).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Fashion-MNIST class names (matching YOLO's label order)
CLASS_NAMES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]

# Dataset root — reuse YOLO's already-downloaded Fashion-MNIST
DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "fashion-mnist"


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Build transforms matching YOLO's classification pipeline:
      - Resize to 28×28 (already the case, but explicit)
      - Convert grayscale PIL → RGB tensor (3 channels)
      - Normalize with ImageNet mean/std (YOLO default for cls)
    """
    base = [
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),  # 1-ch → 3-ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(base)


def get_dataloaders(
    batch_size: int = 16,
    workers: int = 8,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) reading from the YOLO dataset folder.

    The YOLO fashion-mnist directory has:
        datasets/fashion-mnist/train/<class_id>/image.png
        datasets/fashion-mnist/test/<class_id>/image.png
    """
    train_dataset = datasets.ImageFolder(
        root=str(DATASET_ROOT / "train"),
        transform=get_transforms("train"),
    )
    val_dataset = datasets.ImageFolder(
        root=str(DATASET_ROOT / "test"),
        transform=get_transforms("val"),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    return train_loader, val_loader
