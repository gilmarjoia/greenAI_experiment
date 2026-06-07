"""
run.py — Entry point for the YOLO26 Round 5 experiment.

Orchestrates:
  1. Dataset loading   (tracked by CodeCarbon task 'load dataset')
     A custom fashion-mnist.yaml with an explicit absolute path is generated
     at runtime so the download always lands in <project>/datasets/,
     regardless of the global ultralytics settings.
  2. Model training    (tracked by CodeCarbon task 'train model')

Mirrors the YOLO26 baseline experiment configuration but with 20 epochs:
  - 20 epochs, batch=16, default parameters
  - Outputs land in  yolo26_round5/output/runs/train/
  - CodeCarbon log   yolo26_round5/output/emissions.csv
"""

from pathlib import Path
import sys
import os
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

# Allow relative imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent                          # yolo26_round5/
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
        # Official download URL (same as ultralytics built-in)
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
    "model":   f"{DATASETS_DIR}/yolo26n-cls.pt",
    "epochs":  20,
    "batch":   16,
    "imgsz":   28,
    "seed":    0,
    "workers": 8,
    "amp":     True,
}


def main():
    print("=" * 60)
    print("YOLO26 Round 5 — Fashion-MNIST Classification")
    print("=" * 60)
    print(f"Model      : {CONFIG['model']}")
    print(f"Epochs     : {CONFIG['epochs']}")
    print(f"Batch      : {CONFIG['batch']}")
    print(f"ImgSz      : {CONFIG['imgsz']}")
    print(f"Dataset dir: {DATASET_DIR}")

    tracker = EmissionsTracker(
        project_name="yolo_round5_training",
        measure_power_secs=10,
        output_dir=str(OUTPUT_DIR),
    )

    try:
        # ── Task 1: Download dataset ──────────────────────────────────────────
        tracker.start_task("load dataset")
        model = YOLO(CONFIG["model"])
        tracker.stop_task()

        # ── Task 2: Train model ───────────────────────────────────────────────
        tracker.start_task("train model")
        model.train(
            data="fashion-mnist",
            epochs=CONFIG["epochs"],
            batch=CONFIG["batch"],
            imgsz=CONFIG["imgsz"],
            seed=CONFIG["seed"],
            workers=CONFIG["workers"],
            amp=CONFIG["amp"],
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
