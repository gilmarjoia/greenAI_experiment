"""
run.py — Entry point for the Transformers Round 6 experiment.

Orchestrates:
  1. Dataset loading  (tracked by CodeCarbon task 'load dataset')
  2. Model training   (tracked by CodeCarbon task 'train model')
  3. Plot generation  (results.png, confusion matrices, batch images)
  4. Config save      (args.yaml — consumed by evaluation.py)

Round 6 — 30 Epoch Baseline:
  - 30 epochs, batch=16, AdamW lr=1e-4, warmup 3 epochs, cosine decay
  - Same hyperparameters as baseline (Round 1), only epochs increased to 30
  - Outputs land in  transformers_round6/output/runs/train/
  - CodeCarbon log   transformers_round6/output/emissions.csv
"""

import random
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from codecarbon import EmissionsTracker

# Allow relative imports when run as a script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dataset import get_dataloaders
from model import TransformerBaseline
from plots import plot_batch_images, plot_results
from train import train

import platform

# ─── Hyperparameters (baseline — same as Round 1) ────────────────────────────
_IS_WINDOWS = platform.system() == "Windows"
_SAFE_WORKERS = 0 if _IS_WINDOWS else 8

CONFIG = {
    # Data
    "batch": 16,            # ← baseline batch
    "workers": _SAFE_WORKERS,
    "imgsz": 224,          # DeiT-Tiny requires 224×224
    "seed": 0,
    "num_classes": 10,
    # Optimisation (all baseline defaults)
    "epochs": 30,           # ← 30 epochs (Round 6 goal)
    "lr0": 1e-4,           # AdamW fine-tuning LR
    "lrf": 0.01,           # final LR = lrf * lr0
    "weight_decay": 0.05,  # standard AdamW wd for ViT
    "warmup_epochs": 3,
    "amp": True,
    # Model
    "model_id": TransformerBaseline.MODEL_ID,
}

# ─── Output directories ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_DIR = OUTPUT_DIR / "runs" / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True   # auto-tune fastest conv kernels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model        : {CONFIG['model_id']}")
    print(f"Workers      : {CONFIG['workers']} ({'Windows-safe: main-process loading' if _IS_WINDOWS else 'multi-process'})")
    print(f"Epochs       : {CONFIG['epochs']}")

    # Save config for evaluation.py
    args_path = TRAIN_DIR / "args.yaml"
    with open(args_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)
    print(f"Config saved to: {args_path}")

    # ── CodeCarbon tracker ────────────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name="transformers_round6_training",
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
            weight_decay=CONFIG["weight_decay"],
            warmup_epochs=CONFIG["warmup_epochs"],
            amp=CONFIG["amp"],
            device=device,
            num_classes=CONFIG["num_classes"],
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
