"""
evaluation.py — Evaluation script for transformers_modified.

Identical logic to transformers_baseline/evaluation.py.
Only the CodeCarbon project name and directory references are updated.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from codecarbon import EmissionsTracker

from dataset import get_dataloaders
from model import TransformerBaseline
from plots import plot_batch_images, plot_confusion_matrices
from train import run_epoch


def load_config(train_dir: Path) -> dict:
    """Load configuration from the training run."""
    args_path = train_dir / "args.yaml"
    if args_path.exists():
        with open(args_path, "r") as f:
            return yaml.safe_load(f)
    else:
        return {
            "batch":       32,
            "workers":     0,
            "imgsz":       224,
            "seed":        0,
            "num_classes": 10,
        }


def main():
    # ─── Setup ────────────────────────────────────────────────────────────────
    BASE_DIR  = Path(__file__).parent
    TRAIN_DIR = BASE_DIR / "output" / "runs" / "train"
    VAL_DIR   = BASE_DIR / "output" / "runs" / "val"
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(TRAIN_DIR)

    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tracker = EmissionsTracker(
        project_name="transformers_modified_validation",
        measure_power_secs=10,
        output_dir=str(BASE_DIR / "output"),
    )

    try:
        # ── Task 1: Load dataset ──────────────────────────────────────────────
        tracker.start_task("load dataset")
        _, val_loader = get_dataloaders(
            batch_size=config["batch"],
            workers=config["workers"],
            seed=seed,
        )
        print(f"Loaded validation set: {len(val_loader.dataset):,} samples")
        tracker.stop_task()

        # ── Task 2: Validate model ────────────────────────────────────────────
        tracker.start_task("validate model")

        model = TransformerBaseline(num_classes=config["num_classes"]).to(device)

        weights_path = TRAIN_DIR / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"WARNING: Weights not found at {weights_path}. Evaluation might be invalid.")
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded weights from: {weights_path}")

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        val_loss, val_top1, val_top5 = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            training=False,
        )
        tracker.stop_task()

        print(f"\nValidation Results:")
        print(f"  Loss:   {val_loss:.5f}")
        print(f"  Top-1:  {val_top1:.4f}")
        print(f"  Top-5:  {val_top5:.4f}")

        results_path = VAL_DIR / "val_results.csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["metrics/loss", "metrics/accuracy_top1", "metrics/accuracy_top5"],
            )
            writer.writeheader()
            writer.writerow({
                "metrics/loss":             round(val_loss, 5),
                "metrics/accuracy_top1":    round(val_top1, 4),
                "metrics/accuracy_top5":    round(val_top5, 4),
            })

        print("\nGenerating artifacts...")
        plot_confusion_matrices(model, val_loader, device, VAL_DIR)
        plot_batch_images(val_loader, val_loader, model, device, VAL_DIR, n_batches=3)

        print(f"\nAll validation artifacts saved to: {VAL_DIR}")

    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
