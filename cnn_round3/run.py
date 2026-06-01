"""
run.py — Entry point for the CNN Round 3 experiment.

Orchestrates:
  1. Dataset loading  (tracked by CodeCarbon task 'load dataset')
  2. Model training   (tracked by CodeCarbon task 'train model')
  3. Plot generation  (results.png, confusion matrices, batch images)
  4. Config save      (args.yaml — consumed by evaluation.py)

Analysis of Round 2 (cnn_modified) results:
  • Round 2 achieved Top-1 = 93.19% (below baseline 93.44%)
  • Train loss (0.650) ≈ Val loss (0.656) → overfitting eliminated, but model
    underperforms. The dropout=0.3 was too aggressive for this small CNN.
  • Loss curve still slightly descending at epoch 20 — more epochs help.
  • LR schedule bottomed out at 0.0001 — could start slightly higher to maintain
    gradient signal longer.

Hyperparameter changes vs cnn_modified (Round 2):
  ┌─────────────────────┬──────────────┬──────────────┬───────────────────────────────────────────────┐
  │ Parameter           │ Modified     │ Round 3      │ Rationale                                     │
  ├─────────────────────┼──────────────┼──────────────┼───────────────────────────────────────────────┤
  │ epochs              │ 20           │ 30           │ Curve still descending; more steps to converge│
  │ dropout             │ 0.3          │ 0.2          │ 0.3 too aggressive — reduced underfitting     │
  │ lr0                 │ 0.01         │ 0.015        │ Slightly higher peak LR → larger gradient step│
  │ lrf                 │ 0.01         │ 0.005        │ Decays to 0.015*0.005=7.5e-5 — finer tuning  │
  │ weight_decay        │ 0.001        │ 0.0005       │ Less L2 — dropout already provides regulari. │
  │ label_smoothing     │ 0.1          │ 0.1          │ Unchanged — prevents overconfidence           │
  │ warmup_epochs       │ 3            │ 5            │ Longer warmup for higher lr0                  │
  └─────────────────────┴──────────────┴──────────────┴───────────────────────────────────────────────┘

Outputs land in  cnn_round3/output/runs/train/
CodeCarbon log   cnn_round3/output/emissions.csv
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
    "batch":           32,       # unchanged — stable gradient estimates
    "workers":         _SAFE_WORKERS,
    "imgsz":           28,
    "seed":            0,
    "num_classes":     10,
    # Optimisation
    "epochs":          30,       # ↑ from 20 — curve still descending at ep.20
    "lr0":             0.015,    # ↑ from 0.01 — stronger gradient signal
    "lrf":             0.005,    # ↓ from 0.01 — finer cosine decay floor
    "momentum":        0.937,
    "weight_decay":    0.0005,   # ↓ from 0.001 — less L2, dropout handles reg.
    "warmup_epochs":   5,        # ↑ from 3 — longer warmup for higher lr0
    "amp":             True,
    # Regularisation
    "dropout":         0.2,      # ↓ from 0.3 — was too aggressive causing underfit
    "label_smoothing": 0.1,      # unchanged — prevents overconfident softmax outputs
    # Augmentation is applied in dataset.py (flip + rotation + color jitter)
}


# ─── Output directories ───────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_DIR  = OUTPUT_DIR / "runs" / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("CNN Round 3 — Fashion-MNIST Classification")
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
        project_name="cnn_round3_training",
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
