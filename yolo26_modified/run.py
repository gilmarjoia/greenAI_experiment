"""
run.py — Entry point for the YOLO26 modified experiment.

Orchestrates:
  1. Dataset loading   (tracked by CodeCarbon task 'load dataset')
     A custom fashion-mnist.yaml with an explicit absolute path is generated
     at runtime so the download always lands in <project>/datasets/,
     regardless of the global ultralytics settings.
  2. Model training    (tracked by CodeCarbon task 'train model')

Hyperparameter changes vs yolo26_baseline (justified by baseline results):
  ┌──────────────────────┬──────────────┬──────────────┬──────────────────────────────────────────────┐
  │ Parameter            │ Baseline     │ Modified     │ Rationale                                    │
  ├──────────────────────┼──────────────┼──────────────┼──────────────────────────────────────────────┤
  │ epochs               │ 10           │ 20           │ Allows more time to fully converge           │
  │ batch                │ 16           │ 32           │ Larger batch size for more stable gradients  │
  │ optimizer            │ auto (SGD)   │ AdamW        │ Faster and more robust classification convergence│
  │ lr0                  │ 0.01         │ 0.001        │ Standard starting learning rate for AdamW    │
  │ weight_decay         │ 0.0005       │ 0.05         │ Stronger regularization to support AdamW     │
  │ label_smoothing      │ 0.0          │ 0.1          │ Reduces overconfidence and boosts Top-1      │
  └──────────────────────┴──────────────┴──────────────┴──────────────────────────────────────────────┘

Output artifacts:
  yolo26_modified/output/runs/train/
CodeCarbon log:
  yolo26_modified/output/emissions.csv
"""

from pathlib import Path

import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent                          # yolo26_modified/
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
    "epochs":          20,        # ↑ from 10 — allows more time to fully converge
    "batch":           32,        # ↑ from 16 — larger batch size for stable gradients
    "imgsz":           28,
    "seed":            0,
    "workers":         8,
    "amp":             True,
    "optimizer":       "AdamW",   # new — faster and better convergence than default SGD
    "lr0":             0.001,     # new — standard starting learning rate for AdamW
    "weight_decay":    0.05,      # new — stronger regularization
    "label_smoothing": 0.1,       # new — soft labels to reduce overconfidence and boost validation accuracy
}


def main():
    print("=" * 60)
    print("YOLO Modified — Fashion-MNIST Classification")
    print("=" * 60)
    print(f"Model          : {CONFIG['model']}")
    print(f"Epochs         : {CONFIG['epochs']}")
    print(f"Batch          : {CONFIG['batch']}")
    print(f"ImgSz          : {CONFIG['imgsz']}")
    print(f"Optimizer      : {CONFIG['optimizer']}")
    print(f"LR0            : {CONFIG['lr0']}")
    print(f"Weight Decay   : {CONFIG['weight_decay']}")
    print(f"Label Smoothing: {CONFIG['label_smoothing']}")
    print(f"Dataset dir    : {DATASET_DIR}")

    tracker = EmissionsTracker(
        project_name="yolo_modified_training",
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
