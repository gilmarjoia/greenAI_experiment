"""
run.py — Entry point for the YOLO26 Round 3 experiment.

Orchestrates:
  1. Dataset loading   (tracked by CodeCarbon task 'load dataset')
     A custom fashion-mnist.yaml with an explicit absolute path is generated
     at runtime so the download always lands in <project>/datasets/,
     regardless of the global ultralytics settings.
  2. Model training    (tracked by CodeCarbon task 'train model')

Analysis of Round 2 (yolo26_modified) results:
  • Round 2 achieved Top-1 = 91.62% — lowest of the three models.
  • Val loss (0.223) is excellent and still decreasing at epoch 20.
  • Train loss (0.348) vs val loss (0.223) shows minor overfitting signal.
  • The model clearly has more headroom — only limited by number of epochs.
  • LR was already well-tuned (AdamW 0.001). Slight increase helps with 30 epochs.
  • Dropout was disabled; adding a small value can stabilize further tuning.

Hyperparameter changes vs yolo26_modified (Round 2):
  ┌──────────────────────┬──────────────┬──────────────┬──────────────────────────────────────────────────┐
  │ Parameter            │ Modified     │ Round 3      │ Rationale                                        │
  ├──────────────────────┼──────────────┼──────────────┼──────────────────────────────────────────────────┤
  │ epochs               │ 20           │ 30           │ Val loss still descending — 10 more epochs helps │
  │ lr0                  │ 0.001        │ 0.0012       │ Slight increase for more learning with 30 epochs │
  │ lrf                  │ 0.01         │ 0.005        │ Lower floor → finer fine-tuning at end           │
  │ dropout              │ 0.0          │ 0.1          │ Small dropout for stability in later epochs      │
  │ cos_lr               │ false        │ true         │ Cosine LR decay for smoother convergence         │
  │ weight_decay         │ 0.05         │ 0.001        │ Reduce L2 — AdamW + dropout already regularizes  │
  │ label_smoothing      │ 0.1          │ 0.1          │ Unchanged — keeps soft labels                    │
  └──────────────────────┴──────────────┴──────────────┴──────────────────────────────────────────────────┘

Output artifacts:
  yolo26_round3/output/runs/train/
CodeCarbon log:
  yolo26_round3/output/emissions.csv
"""

from pathlib import Path

import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent                          # yolo26_round3/
PROJECT_ROOT = BASE_DIR.parent                                # greenAI_experiment/
DATASETS_DIR = PROJECT_ROOT / "datasets"                      # greenAI_experiment/datasets/
DATASET_DIR  = DATASETS_DIR / "fashion-mnist"                 # …/datasets/fashion-mnist/
OUTPUT_DIR   = BASE_DIR / "output"
TRAIN_DIR    = OUTPUT_DIR / "runs" / "train"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Custom dataset YAML (absolute path — bypasses global settings cache) ─────
_YAML_PATH = DATASETS_DIR / "fashion-mnist.yaml"

def _write_dataset_yaml() -> None:
    """Write (or overwrite) a custom fashion-mnist.yaml with the absolute path."""
    config = {
        "path": str(DATASET_DIR),   # absolute — no ambiguity
        "train": "train",
        "val":   "test",
        "nc":    10,
        "names": {
            0: "T-shirt/top", 1: "Trouser",   2: "Pullover",
            3: "Dress",       4: "Coat",       5: "Sandal",
            6: "Shirt",       7: "Sneaker",    8: "Bag",
            9: "Ankle boot",
        },
        "download": (
            "https://github.com/ultralytics/assets/releases/"
            "download/v0.0.0/fashion-mnist.zip"
        ),
    }
    with open(_YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Dataset YAML written → {_YAML_PATH}")


# ─── Hyperparameters ──────────────────────────────────────────────────────────
CONFIG = {
    "model":           f"{DATASETS_DIR}/yolo26n-cls.pt",
    "epochs":          30,        # ↑ from 20 — val loss still descending at ep.20
    "batch":           32,        # unchanged — already optimal batch size
    "imgsz":           28,
    "seed":            0,
    "workers":         8,
    "amp":             True,
    "optimizer":       "AdamW",   # unchanged — best optimizer for this task
    "lr0":             0.0012,    # ↑ from 0.001 — slightly higher for 30 epochs
    "lrf":             0.005,     # ↓ from 0.01 — finer decay floor at end of training
    "cos_lr":          True,      # new — cosine LR decay for smoother convergence
    "dropout":         0.1,       # new — small dropout to stabilize late epochs
    "weight_decay":    0.001,     # ↓ from 0.05 — AdamW + dropout enough to regularize
    "label_smoothing": 0.1,       # unchanged — soft labels reduce overconfidence
}


def main():
    print("=" * 60)
    print("YOLO26 Round 3 — Fashion-MNIST Classification")
    print("=" * 60)
    print(f"Model          : {CONFIG['model']}")
    print(f"Epochs         : {CONFIG['epochs']}")
    print(f"Batch          : {CONFIG['batch']}")
    print(f"ImgSz          : {CONFIG['imgsz']}")
    print(f"Optimizer      : {CONFIG['optimizer']}")
    print(f"LR0            : {CONFIG['lr0']}")
    print(f"LRf            : {CONFIG['lrf']}")
    print(f"Cosine LR      : {CONFIG['cos_lr']}")
    print(f"Dropout        : {CONFIG['dropout']}")
    print(f"Weight Decay   : {CONFIG['weight_decay']}")
    print(f"Label Smoothing: {CONFIG['label_smoothing']}")
    print(f"Dataset dir    : {DATASET_DIR}")

    tracker = EmissionsTracker(
        project_name="yolo_round3_training",
        measure_power_secs=10,
        output_dir=str(OUTPUT_DIR),
    )

    try:
        # ── Task 1: Load model ────────────────────────────────────────────────
        tracker.start_task("load dataset")
        model = YOLO(CONFIG["model"])
        tracker.stop_task()

        # ── Task 2: Train model ───────────────────────────────────────────────
        tracker.start_task("train model")
        model.train(
            data=f"fashion-mnist",
            epochs=CONFIG["epochs"],
            batch=CONFIG["batch"],
            imgsz=CONFIG["imgsz"],
            seed=CONFIG["seed"],
            workers=CONFIG["workers"],
            amp=CONFIG["amp"],
            optimizer=CONFIG["optimizer"],
            lr0=CONFIG["lr0"],
            lrf=CONFIG["lrf"],
            cos_lr=CONFIG["cos_lr"],
            dropout=CONFIG["dropout"],
            weight_decay=CONFIG["weight_decay"],
            label_smoothing=CONFIG["label_smoothing"],
            project=str(OUTPUT_DIR / "runs"),
            name="train",
            exist_ok=True,
        )
        tracker.stop_task()

        print(f"\nAll training artifacts saved to: {TRAIN_DIR}")

    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
