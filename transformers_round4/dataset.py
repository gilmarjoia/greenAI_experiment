"""
dataset.py — Fashion-MNIST loader for transformers_round4.

Identical to transformers_round3/dataset.py — reuses the same augmentation
pipeline (RandomHorizontalFlip, RandomRotation(15°), ColorJitter, RandomAffine)
that successfully closed the train/val gap in Round 2.

Changes vs transformers_baseline:
  - Training split: adds RandomRotation(15°), ColorJitter, RandomAffine on top
    of the existing RandomHorizontalFlip.
  - Validation split: unchanged (deterministic preprocessing only).
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

# DeiT-Tiny requires 224×224 inputs
IMG_SIZE = 224


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Build transforms for transformers_modified:
      - Resize to 224×224 (ViT patch size requirement)
      - Convert grayscale PIL → RGB tensor (3 channels)
      - Stronger augmentation on train split to close the train/val loss gap
      - Normalize with ImageNet mean/std (matching pretrained backbone)

    Augmentation additions vs baseline:
      - RandomRotation(15°): small tilts common in real clothing photos.
      - ColorJitter(brightness, contrast): grayscale images replicated to 3ch
        benefit from intensity diversity.
      - RandomAffine(shear=10): mild shear adds invariance to viewpoint shift.
    """
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # 1-ch → 3-ch
    ]

    if split == "train":
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, shear=10),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(base)


def get_dataloaders(
    batch_size: int = 32,
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
