"""
plots.py — Generate training artifacts equivalent to YOLO's output:
  - results.png       : loss + accuracy curves across epochs
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - train_batch{0,1,2}.jpg
  - val_batch{0,1,2}_labels.jpg
  - val_batch{0,1,2}_pred.jpg
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from dataset import CLASS_NAMES


# ─── Style constants (matching a clean, publication-style look) ───────────────
COLORS = {
    "train": "#4C72B0",
    "val": "#DD8452",
}
FIG_DPI = 100


# ─── Results curves ──────────────────────────────────────────────────────────

def plot_results(results: list[dict], save_dir: Path) -> None:
    """
    4-panel plot (loss train, loss val, top1 acc, top5 acc) across epochs.
    Matches YOLO's results.png layout.
    """
    epochs = [r["epoch"] for r in results]
    train_loss = [r["train/loss"] for r in results]
    val_loss = [r["val/loss"] for r in results]
    top1 = [r["metrics/accuracy_top1"] for r in results]
    top5 = [r["metrics/accuracy_top5"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("CNN Round 5 — Training Results", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], "Train Loss",       train_loss, COLORS["train"]),
        (axes[0, 1], "Val Loss",          val_loss,   COLORS["val"]),
        (axes[1, 0], "Top-1 Accuracy",   top1,       COLORS["train"]),
        (axes[1, 1], "Top-5 Accuracy",   top5,       COLORS["val"]),
    ]

    for ax, title, data, color in panels:
        ax.plot(epochs, data, color=color, linewidth=2, marker="o", markersize=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = save_dir / "results.png"
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def _compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            for t, p in zip(labels, preds):
                matrix[t, p] += 1
    return matrix


def _plot_cm(matrix: np.ndarray, save_path: Path, normalized: bool = False) -> None:
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        cm = matrix
        fmt = "d"
        title = "Confusion Matrix"

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm[i, j]:{fmt}}" if fmt == "d" else f"{cm[i, j]:.2f}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_matrices(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
) -> None:
    matrix = _compute_confusion_matrix(model, val_loader, device)
    _plot_cm(matrix, save_dir / "confusion_matrix.png", normalized=False)
    _plot_cm(matrix, save_dir / "confusion_matrix_normalized.png", normalized=True)


# ─── Batch preview images ─────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _save_batch_image(
    images: torch.Tensor,
    labels: list[int],
    preds: list[int] | None,
    save_path: Path,
    class_names: list[str],
) -> None:
    """Save a grid of images with true/predicted labels as a JPEG."""
    n = min(len(images), 16)
    images = images[:n]
    imgs_denorm = torch.stack([_denormalize(img.cpu()) for img in images])
    grid = make_grid(imgs_denorm, nrow=4, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    axes.imshow(grid_np)
    axes.axis("off")

    # Build label text
    label_text = []
    for i in range(n):
        true_name = class_names[labels[i]]
        if preds is not None:
            pred_name = class_names[preds[i]]
            label_text.append(f"{true_name}→{pred_name}")
        else:
            label_text.append(true_name)
    axes.set_title("\n".join(
        "  ".join(label_text[i:i+4]) for i in range(0, len(label_text), 4)
    ), fontsize=6, family="monospace")

    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=FIG_DPI, format="jpeg")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_batch_images(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    save_dir: Path,
    class_names: list[str] = CLASS_NAMES,
    n_batches: int = 3,
) -> None:
    """
    Generate train_batch{i}.jpg and val_batch{i}_labels/pred.jpg files.
    Mirrors the YOLO artifact naming convention.
    """
    model.eval()

    # Training batches (labels only — no prediction overlay)
    train_iter = iter(train_loader)
    for i in range(n_batches):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            break
        _save_batch_image(
            images, labels.tolist(), None,
            save_dir / f"train_batch{i}.jpg",
            class_names,
        )

    # Validation batches (labels + predictions)
    val_iter = iter(val_loader)
    for i in range(n_batches):
        try:
            images, labels = next(val_iter)
        except StopIteration:
            break
        with torch.no_grad():
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().tolist()

        _save_batch_image(
            images, labels.tolist(), None,
            save_dir / f"val_batch{i}_labels.jpg",
            class_names,
        )
        _save_batch_image(
            images, labels.tolist(), preds,
            save_dir / f"val_batch{i}_pred.jpg",
            class_names,
        )
