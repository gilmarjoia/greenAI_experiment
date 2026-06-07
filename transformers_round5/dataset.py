"""
dataset.py — Fashion-MNIST loader for the ViT/Transformers baseline.

Reuses the YOLO-format directory (subfolders 0–9), identical to the CNN baseline.
Images are resized to 224×224 to match ViT-B/16's expected patch size,
and normalised with ImageNet stats (same as the pretrained backbone).
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

# ViT-B/16 expects 224×224 inputs
IMG_SIZE = 224


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Build transforms suitable for ViT-B/16:
      - Resize to 224×224 (ViT patch size requirement)
      - Convert grayscale PIL → RGB tensor (3 channels)
      - Light augmentation on train split only (random horizontal flip)
      - Normalize with ImageNet mean/std (matching pretrained backbone)
    """
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # 1-ch → 3-ch
    ]

    if split == "train":
        base.append(transforms.RandomHorizontalFlip())

    base += [
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
