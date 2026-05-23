"""
run.py — Entry point for the Transformers modified experiment.

Orchestrates:
  1. Dataset loading  (tracked by CodeCarbon task 'load dataset')
  2. Model training   (tracked by CodeCarbon task 'train model')
  3. Plot generation  (results.png, confusion matrices, batch images)
  4. Config save      (args.yaml — consumed by evaluation.py)

Hyperparameter changes vs transformers_baseline (justified by baseline results):
  ┌──────────────────────┬──────────────┬──────────────┬──────────────────────────────────────────────┐
  │ Parameter            │ Baseline     │ Modified     │ Rationale                                    │
  ├──────────────────────┼──────────────┼──────────────┼──────────────────────────────────────────────┤
  │ epochs               │ 10           │ 20           │ ViT fine-tuning improves steadily past ep.10 │
  │ batch                │ 16           │ 32           │ Larger batches → stabler attention gradients │
  │ lr0                  │ 1e-4         │ 5e-5         │ Lower LR → more careful weight updates       │
  │ lrf                  │ 0.01         │ 0.01         │ Unchanged — cosine decay ratio               │
  │ weight_decay         │ 0.05         │ 0.1          │ Stronger AdamW regularisation                │
  │ label_smoothing      │ —            │ 0.1          │ Reduces overconfident softmax outputs        │
  │ gradient clipping    │ —            │ max_norm=1.0 │ Stabilises ViT training                      │
  │ data augmentation    │ flip only    │ +rot+jit+shear│ Closes the 0.021 train vs 0.184 val gap    │
  └──────────────────────┴──────────────┴──────────────┴──────────────────────────────────────────────┘

Outputs land in  transformers_modified/output/runs/train/
CodeCarbon log   transformers_modified/output/emissions.csv
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
from model import TransformerBaseline
from plots import plot_batch_images, plot_results
from train import train

_IS_WINDOWS   = platform.system() == "Windows"
_SAFE_WORKERS = 0 if _IS_WINDOWS else 8

# ─── Hyperparameters ─────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "batch":            32,        # ↑ from 16 — stabler attention-layer gradients
    "workers":          _SAFE_WORKERS,
    "imgsz":            224,       # DeiT-Tiny requires 224×224 — unchanged
    "seed":             0,
    "num_classes":      10,
    # Optimisation
    "epochs":           20,        # ↑ from 10 — ViT fine-tuning improves steadily
    "lr0":              5e-5,      # ↓ from 1e-4 — more conservative weight updates
    "lrf":              0.01,      # final LR = lrf * lr0
    "weight_decay":     0.1,       # ↑ from 0.05 — stronger AdamW regularisation
    "warmup_epochs":    3,
    "amp":              True,
    # Regularisation (new in modified)
    "label_smoothing":  0.1,       # new — prevents overconfident softmax outputs
    "grad_clip_norm":   1.0,       # new — stabilises ViT fine-tuning
    # Model
    "model_id":         TransformerBaseline.MODEL_ID,
    # Augmentation is applied in dataset.py (flip + rotation + jitter + shear)
}

# ─── Output directories ───────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_DIR  = OUTPUT_DIR / "runs" / "train"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.benchmark = True   # auto-tune fastest conv kernels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device     : {device}")
    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"VRAM total       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model            : {CONFIG['model_id']}")
    print(f"Epochs           : {CONFIG['epochs']}")
    print(f"Batch            : {CONFIG['batch']}")
    print(f"LR0              : {CONFIG['lr0']}")
    print(f"Weight decay     : {CONFIG['weight_decay']}")
    print(f"Label smoothing  : {CONFIG['label_smoothing']}")
    print(f"Grad clip norm   : {CONFIG['grad_clip_norm']}")
    print(f"Augmentation     : flip + rotation(15°) + ColorJitter + shear(10°)")
    print(f"Workers          : {CONFIG['workers']} ({'Windows-safe: main-process' if _IS_WINDOWS else 'multi-process'})")

    # Save config for evaluation.py
    args_path = TRAIN_DIR / "args.yaml"
    with open(args_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False)
    print(f"Config saved to  : {args_path}")

    tracker = EmissionsTracker(
        project_name="transformers_modified_training",
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
            label_smoothing=CONFIG["label_smoothing"],
            grad_clip_norm=CONFIG["grad_clip_norm"],
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
