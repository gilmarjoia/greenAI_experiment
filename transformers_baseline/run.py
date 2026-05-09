"""
run.py — Entry point for the ViT/Transformers baseline experiment.

Orchestrates:
  1. Dataset loading  (tracked by CodeCarbon task 'load dataset')
  2. Model training   (tracked by CodeCarbon task 'train model')
  3. Plot generation  (results.png, confusion matrices, batch images)
  4. Config save      (args.yaml — consumed by evaluation.py)

Mirrors the YOLO26 / CNN baseline experiment structure exactly:
  - 10 epochs, batch=16, AdamW lr=1e-4, warmup 3 epochs, cosine decay
  - Outputs land in  transformers_baseline/output/runs/train/
  - CodeCarbon log   transformers_baseline/output/emissions.csv
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from codecarbon import EmissionsTracker

# Allow relative imports when run as a script
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import ViTBaseline
from plots import plot_batch_images, plot_results
from train import train


# ─── Hyperparameters (mirror CNN / YOLO26 where applicable) ──────────────────
CONFIG = {
    # Data
    "batch": 16,
    "workers": 8,
    "imgsz": 224,          # ViT-B/16 requires 224×224
    "seed": 0,
    "num_classes": 10,
    # Optimisation
    "epochs": 10,
    "lr0": 1e-4,           # AdamW fine-tuning LR (vs 0.01 SGD for CNN)
    "lrf": 0.01,           # final LR = lrf * lr0
    "weight_decay": 0.05,  # standard AdamW wd for ViT
    "warmup_epochs": 3,
    "amp": True,
    # Model
    "model_id": ViTBaseline.MODEL_ID,
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
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {CONFIG['model_id']}")

    # Save config for evaluation.py
    args_path = TRAIN_DIR / "args.yaml"
    with open(args_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)
    print(f"Config saved to: {args_path}")

    # ── CodeCarbon tracker ────────────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name="transformers_baseline_training",
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
