"""
run.py — Entry point for the CNN Round 4 experiment.

Orchestrates:
  1. Dataset loading  (tracked by CodeCarbon task 'load dataset')
  2. Model training   (tracked by CodeCarbon task 'train model')
  3. Plot generation  (results.png, confusion matrices, batch images)
  4. Config save      (args.yaml — consumed by evaluation.py)

Goal of Round 4 — Efficiency benchmark:
  Reproduce the full Round 3 hyperparameter stack (all regularisation and
  augmentation improvements) but train for only 10 epochs.

  This allows a direct comparison with the Baseline (also 10 epochs, raw
  hyperparameters) to isolate the effect of the improved hyperparameter set
  from the effect of more training time.

  Hyperparameters vs cnn_round3:
  ┌─────────────────────┬──────────────┬──────────────┬─────────────────────────────────────────────────────┐
  │ Parameter           │ Round 3      │ Round 4      │ Rationale                                           │
  ├─────────────────────┼──────────────┼──────────────┼─────────────────────────────────────────────────────┤
  │ epochs              │ 30           │ 10           │ Efficiency benchmark — same epoch count as Baseline │
  │ dropout             │ 0.2          │ 0.2          │ Unchanged                                           │
  │ lr0                 │ 0.015        │ 0.015        │ Unchanged                                           │
  │ lrf                 │ 0.005        │ 0.005        │ Unchanged                                           │
  │ weight_decay        │ 0.0005       │ 0.0005       │ Unchanged                                           │
  │ label_smoothing     │ 0.1          │ 0.1          │ Unchanged                                           │
  │ warmup_epochs       │ 5            │ 3            │ Reduced proportionally (3/10 ≈ 5/30)               │
  └─────────────────────┴──────────────┴──────────────┴─────────────────────────────────────────────────────┘

Outputs land in  cnn_round4/output/runs/train/
CodeCarbon log   cnn_round4/output/emissions.csv
"""

import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from codecarbon import EmissionsTracker

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from plots import plot_batch_images, plot_results
from train import train

_IS_WINDOWS   = platform.system() == "Windows"
_SAFE_WORKERS = 0 if _IS_WINDOWS else 8

# ─── Hyperparameters ─────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "batch":           32,       # unchanged from Round 3
    "workers":         _SAFE_WORKERS,
    "imgsz":           28,
    "seed":            0,
    "num_classes":     10,
    # Optimisation
    "epochs":          10,       # ↓ from 30 — efficiency benchmark (= Baseline epoch count)
    "lr0":             0.015,    # unchanged from Round 3
    "lrf":             0.005,    # unchanged from Round 3
    "momentum":        0.937,
    "weight_decay":    0.0005,   # unchanged from Round 3
    "warmup_epochs":   3,        # ↓ from 5 — proportional reduction for 10 epochs
    "amp":             True,
    # Regularisation
    "dropout":         0.2,      # unchanged from Round 3
    "label_smoothing": 0.1,      # unchanged from Round 3
    # Augmentation is applied in dataset.py (flip + rotation + color jitter)
}


# ─── Output directories ───────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_DIR  = OUTPUT_DIR / "runs" / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("CNN Round 4 — Fashion-MNIST Classification")
    print("=" * 60)
    print(f"Epochs          : {CONFIG['epochs']}")
    print(f"Batch           : {CONFIG['batch']}")
    print(f"LR0             : {CONFIG['lr0']}")
    print(f"LRf             : {CONFIG['lrf']}")
    print(f"Dropout         : {CONFIG['dropout']}")
    print(f"Label smoothing : {CONFIG['label_smoothing']}")
    print(f"Weight decay    : {CONFIG['weight_decay']}")
    print(f"Warmup epochs   : {CONFIG['warmup_epochs']}")
    print(f"Augmentation    : flip + rotation(10°) + ColorJitter")

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    # Save config for evaluation.py
    args_path = TRAIN_DIR / "args.yaml"
    with open(args_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)
    print(f"Config saved to : {args_path}")

    tracker = EmissionsTracker(
        project_name="cnn_round4_training",
        measure_power_secs=10,
        output_dir=str(OUTPUT_DIR),
    )

    try:
        # ── Task 1: Load dataset ──────────────────────────────────────────────
        tracker.start_task("load dataset")
        train_loader, val_loader = get_dataloaders(
            batch_size=CONFIG["batch"],
            workers=CONFIG["workers"],
            seed=CONFIG["seed"],
        )
        print(f"Train samples : {len(train_loader.dataset):,}")
        print(f"Val   samples : {len(val_loader.dataset):,}")
        tracker.stop_task()

        # ── Task 2: Train model ───────────────────────────────────────────────
        tracker.start_task("train model")
        results, model = train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=TRAIN_DIR,
            epochs=CONFIG["epochs"],
            lr0=CONFIG["lr0"],
            lrf=CONFIG["lrf"],
            momentum=CONFIG["momentum"],
            weight_decay=CONFIG["weight_decay"],
            warmup_epochs=CONFIG["warmup_epochs"],
            amp=CONFIG["amp"],
            device=device,
            num_classes=CONFIG["num_classes"],
            dropout=CONFIG["dropout"],
            label_smoothing=CONFIG["label_smoothing"],
        )
        tracker.stop_task()

        # ── Generate plots ────────────────────────────────────────────────────
        print("\nGenerating plots...")
        plot_results(results, TRAIN_DIR)
        plot_batch_images(train_loader, val_loader, model, device, TRAIN_DIR, n_batches=3)

        print(f"\nAll training artifacts saved to: {TRAIN_DIR}")

    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
