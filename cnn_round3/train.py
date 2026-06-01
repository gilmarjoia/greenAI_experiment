"""
train.py — Training loop for cnn_round3.

Changes vs cnn_modified (Round 2):
  - Dropout reduced from 0.3 → 0.2 (passed via run.py CONFIG) — 0.3 caused underfitting.
  - LR0 raised from 0.01 → 0.015 with lrf=0.005 — more gradient signal across more epochs.
  - CrossEntropyLoss label_smoothing=0.1 retained for stability.
  - All other logic (SGD + Nesterov, cosine LR decay, AMP, CSV logging) unchanged.
"""

import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from model import SimpleCNN


# ─── Scheduler ────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs: int, warmup_epochs: int, lrf: float):
    """
    Replicates YOLO's LR schedule:
      - Linear warmup from 0 → lr0 over `warmup_epochs`
      - Cosine decay from lr0 → lrf*lr0 over the remaining epochs
    Returns a LambdaLR scheduler (applied per-epoch).
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lrf + (1.0 - lrf) * cosine               # cosine to lrf

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Accuracy helpers ─────────────────────────────────────────────────────────

def topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """Compute top-k accuracy for the given k values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res


# ─── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    training: bool,
) -> tuple[float, float, float]:
    """Run one epoch. Returns (avg_loss, top1_acc, top5_acc)."""
    model.train(training)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_batches = len(loader)

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            use_amp = scaler is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            top1, top5 = topk_accuracy(outputs.detach(), labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5

    return total_loss / n_batches, total_top1 / n_batches, total_top5 / n_batches


# ─── Main train function ──────────────────────────────────────────────────────

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: Path,
    epochs: int = 20,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.001,
    warmup_epochs: int = 3,
    amp: bool = True,
    device: torch.device | None = None,
    num_classes: int = 10,
    dropout: float = 0.3,
    label_smoothing: float = 0.1,
) -> list[dict]:
    """
    Full training run. Returns (list of per-epoch metric dicts, trained model).
    Saves weights/best.pt and weights/last.pt.

    Key changes vs cnn_baseline:
      - label_smoothing=0.1 passed to CrossEntropyLoss (reduces overconfidence)
      - dropout=0.3 (set in run.py CONFIG, passed here)
      - weight_decay=0.001 (slightly higher than baseline 0.0005)
      - epochs=20 (double the baseline)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=num_classes, dropout=dropout).to(device)

    # Label smoothing regularises the classifier, preventing overconfident
    # predictions that contributed to the train/val loss gap in cnn_baseline.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs, lrf)
    scaler = GradScaler() if (amp and device.type == "cuda") else None

    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / "results.csv"
    fieldnames = [
        "epoch", "time",
        "train/loss",
        "metrics/accuracy_top1", "metrics/accuracy_top5",
        "val/loss",
        "lr/pg0",
    ]

    best_top1 = 0.0
    run_start = time.time()
    all_results = []

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss, train_top1, train_top5 = run_epoch(
                model, train_loader, criterion, optimizer, device, scaler, training=True
            )
            val_loss, val_top1, val_top5 = run_epoch(
                model, val_loader, criterion, None, device, scaler, training=False
            )
            scheduler.step()

            elapsed = time.time() - run_start
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "time": round(elapsed, 3),
                "train/loss": round(train_loss, 5),
                "metrics/accuracy_top1": round(val_top1, 4),
                "metrics/accuracy_top5": round(val_top5, 4),
                "val/loss": round(val_loss, 5),
                "lr/pg0": round(current_lr, 9),
            }
            writer.writerow(row)
            f.flush()
            all_results.append(row)

            print(
                f"Epoch {epoch:>2}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"top1={val_top1:.4f}  "
                f"top5={val_top5:.4f}  "
                f"lr={current_lr:.6f}  "
                f"time={elapsed:.1f}s"
            )

            torch.save(model.state_dict(), weights_dir / "last.pt")

            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save(model.state_dict(), weights_dir / "best.pt")

    print(f"\nTraining complete. Best top-1: {best_top1:.4f}")
    return all_results, model
