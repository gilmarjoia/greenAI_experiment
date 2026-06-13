"""
generate_final_report.py — Final consolidated comparison report generator.

Reads summary metrics from all six experimental phases:
  - Baseline      (reports/summary_metrics.csv)
  - Modified/R2   (reports/summary_metrics_modified.csv)
  - Round 3       (reports/summary_metrics_round3.csv)
  - Round 4       (reports/summary_metrics_round4.csv)
  - Round 5       (reports/summary_metrics_round5.csv)
  - Round 6       (reports/summary_metrics_round6.csv)

Produces:
  - reports/final_report.md          — Full Markdown report (6-phase comparison)
  - reports/final_dashboard.png      — Multi-panel comparison visualization

Usage:
    python generate_final_report.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_BASELINE = REPORTS_DIR / "summary_metrics.csv"
CSV_ROUND2   = REPORTS_DIR / "summary_metrics_modified.csv"
CSV_ROUND3   = REPORTS_DIR / "summary_metrics_round3.csv"
CSV_ROUND4   = REPORTS_DIR / "summary_metrics_round4.csv"
CSV_ROUND5   = REPORTS_DIR / "summary_metrics_round5.csv"
CSV_ROUND6   = REPORTS_DIR / "summary_metrics_round6.csv"
MD_REPORT    = REPORTS_DIR / "final_report.md"
PNG_DASH     = REPORTS_DIR / "final_dashboard.png"

# ─── Hyperparameter tables (all 5 rounds) ─────────────────────────────────────
HYPERPARAMS = {
    "YOLO26": {
        "baseline": (
            "- **Epochs**: 10\n"
            "- **Batch**: 16\n"
            "- **Optimizer**: Auto (SGD)\n"
            "- **LR0**: 0.01\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Dropout**: 0.0\n"
            "- **Label Smoothing**: 0.0"
        ),
        "round2": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 32 (↑)\n"
            "- **Optimizer**: AdamW (change)\n"
            "- **LR0**: 0.001 (↓)\n"
            "- **Weight Decay**: 0.05 (↑)\n"
            "- **Dropout**: 0.0\n"
            "- **Label Smoothing**: 0.1 (↑)"
        ),
        "round3": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 32\n"
            "- **Optimizer**: AdamW\n"
            "- **LR0**: 0.0012 (↑)\n"
            "- **LRf**: 0.005 (↓)\n"
            "- **Weight Decay**: 0.001 (↓)\n"
            "- **Dropout**: 0.1 (↑)\n"
            "- **Cosine LR**: True (new)\n"
            "- **Label Smoothing**: 0.1"
        ),
        "round4": (
            "- **Epochs**: 10 (↓)\n"
            "- **Batch**: 32\n"
            "- **Optimizer**: AdamW\n"
            "- **LR0**: 0.0012\n"
            "- **LRf**: 0.005\n"
            "- **Weight Decay**: 0.001\n"
            "- **Dropout**: 0.1\n"
            "- **Cosine LR**: True\n"
            "- **Label Smoothing**: 0.1"
        ),
        "round5": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 16\n"
            "- **Optimizer**: Auto (SGD)\n"
            "- **LR0**: 0.01\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Dropout**: 0.0\n"
            "- **Label Smoothing**: 0.0"
        ),
        "round6": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 16\n"
            "- **Optimizer**: Auto (SGD)\n"
            "- **LR0**: 0.01\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Dropout**: 0.0\n"
            "- **Label Smoothing**: 0.0"
        ),
    },
    "CNN": {
        "baseline": (
            "- **Epochs**: 10\n"
            "- **Batch**: 16\n"
            "- **LR0**: 0.01\n"
            "- **Dropout**: 0.0\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Label Smoothing**: None\n"
            "- **Augmentation**: None"
        ),
        "round2": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 32 (↑)\n"
            "- **LR0**: 0.01\n"
            "- **Dropout**: 0.3 (↑)\n"
            "- **Weight Decay**: 0.001 (↑)\n"
            "- **Label Smoothing**: 0.1 (new)\n"
            "- **Augmentation**: Flip+Rot(10°)+Jitter (new)"
        ),
        "round3": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 32\n"
            "- **LR0**: 0.015 (↑)\n"
            "- **LRf**: 0.005 (↓)\n"
            "- **Dropout**: 0.2 (↓)\n"
            "- **Weight Decay**: 0.0005 (↓)\n"
            "- **Warmup**: 5 epochs (↑)\n"
            "- **Label Smoothing**: 0.1\n"
            "- **Augmentation**: Flip+Rot(10°)+Jitter"
        ),
        "round4": (
            "- **Epochs**: 10 (↓)\n"
            "- **Batch**: 32\n"
            "- **LR0**: 0.015\n"
            "- **LRf**: 0.005\n"
            "- **Dropout**: 0.2\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Warmup**: 3 epochs (↓)\n"
            "- **Label Smoothing**: 0.1\n"
            "- **Augmentation**: Flip+Rot(10°)+Jitter"
        ),
        "round5": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 16\n"
            "- **LR0**: 0.01\n"
            "- **Dropout**: 0.0\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Label Smoothing**: None\n"
            "- **Warmup**: 3 epochs\n"
            "- **Augmentation**: None"
        ),
        "round6": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 16\n"
            "- **LR0**: 0.01\n"
            "- **Dropout**: 0.0\n"
            "- **Weight Decay**: 0.0005\n"
            "- **Label Smoothing**: None\n"
            "- **Warmup**: 3 epochs\n"
            "- **Augmentation**: None"
        ),
    },
    "ViT": {
        "baseline": (
            "- **Epochs**: 10\n"
            "- **Batch**: 16\n"
            "- **LR0**: 1e-4\n"
            "- **Weight Decay**: 0.05\n"
            "- **Label Smoothing**: None\n"
            "- **Grad Clipping**: None\n"
            "- **Augmentation**: Flip only"
        ),
        "round2": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 32 (↑)\n"
            "- **LR0**: 5e-5 (↓)\n"
            "- **Weight Decay**: 0.1 (↑)\n"
            "- **Label Smoothing**: 0.1 (new)\n"
            "- **Grad Clipping**: max_norm=1.0 (new)\n"
            "- **Augmentation**: +Rot(15°)+Jitter+Shear (↑)"
        ),
        "round3": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 32\n"
            "- **LR0**: 8e-5 (↑)\n"
            "- **LRf**: 0.02 (↑)\n"
            "- **Weight Decay**: 0.05 (↓)\n"
            "- **Warmup**: 5 epochs (↑)\n"
            "- **Label Smoothing**: 0.05 (↓)\n"
            "- **Grad Clipping**: max_norm=1.0\n"
            "- **Augmentation**: Flip+Rot(15°)+Jitter+Shear"
        ),
        "round4": (
            "- **Epochs**: 10 (↓)\n"
            "- **Batch**: 32\n"
            "- **LR0**: 8e-5\n"
            "- **LRf**: 0.02\n"
            "- **Weight Decay**: 0.05\n"
            "- **Warmup**: 2 epochs (↓)\n"
            "- **Label Smoothing**: 0.05\n"
            "- **Grad Clipping**: max_norm=1.0\n"
            "- **Augmentation**: Flip+Rot(15°)+Jitter+Shear"
        ),
        "round5": (
            "- **Epochs**: 20 (↑)\n"
            "- **Batch**: 16\n"
            "- **LR0**: 1e-4\n"
            "- **Weight Decay**: 0.05\n"
            "- **Warmup**: 3 epochs\n"
            "- **Label Smoothing**: None\n"
            "- **Grad Clipping**: None\n"
            "- **Augmentation**: Flip only"
        ),
        "round6": (
            "- **Epochs**: 30 (↑)\n"
            "- **Batch**: 16\n"
            "- **LR0**: 1e-4\n"
            "- **Weight Decay**: 0.05\n"
            "- **Warmup**: 3 epochs\n"
            "- **Label Smoothing**: None\n"
            "- **Grad Clipping**: None\n"
            "- **Augmentation**: Flip only"
        ),
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_csv(path: Path, key_col: str) -> dict:
    """Read CSV into dict keyed by model name."""
    data = {}
    if not path.exists():
        return data
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row[key_col]] = row
    return data


def fv(val, dec=2, suffix=""):
    """Format float value or return N/A."""
    try:
        return f"{float(val):.{dec}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def delta(new_val, old_val, dec=2, suffix=""):
    """Format signed delta between two values."""
    try:
        d = float(new_val) - float(old_val)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{dec}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def pct_delta(new_val, old_val, dec=1):
    """Format percentage delta relative to old value."""
    try:
        n, o = float(new_val), float(old_val)
        if o == 0:
            return "N/A"
        d = (n - o) / o * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{dec}f}%"
    except (ValueError, TypeError):
        return "N/A"


# ─── Model key mappings ───────────────────────────────────────────────────────
MODEL_MAP = [
    {
        "id":       "YOLO26",
        "base_key": "YOLO26",
        "r2_key":   "YOLO26 Modified",
        "r3_key":   "YOLO26 Round 3",
        "r4_key":   "YOLO26 Round 4",
        "r5_key":   "YOLO26 Round 5",
        "r6_key":   "YOLO26 Round 6",
        "label":    "YOLO26",
    },
    {
        "id":       "CNN",
        "base_key": "CNN",
        "r2_key":   "CNN Modified",
        "r3_key":   "CNN Round 3",
        "r4_key":   "CNN Round 4",
        "r5_key":   "CNN Round 5",
        "r6_key":   "CNN Round 6",
        "label":    "CNN",
    },
    {
        "id":       "ViT",
        "base_key": "ViT (Transformers)",
        "r2_key":   "ViT (Transformers) Modified",
        "r3_key":   "ViT (Transformers) Round 3",
        "r4_key":   "ViT (Transformers) Round 4",
        "r5_key":   "ViT (Transformers) Round 5",
        "r6_key":   "ViT (Transformers) Round 6",
        "label":    "ViT",
    },
]


# ─── Plot ─────────────────────────────────────────────────────────────────────

def make_dashboard(baselines, round2s, round3s, round4s, round5s, round6s) -> None:
    """Generate a 1x3 comparison dashboard PNG with 6 bars per model."""
    models = [m["label"] for m in MODEL_MAP]

    acc_b,  acc_r2,  acc_r3,  acc_r4,  acc_r5,  acc_r6   = [], [], [], [], [], []
    em_b,   em_r2,   em_r3,   em_r4,   em_r5,   em_r6    = [], [], [], [], [], []
    time_b, time_r2, time_r3, time_r4, time_r5, time_r6  = [], [], [], [], [], []

    for m in MODEL_MAP:
        b  = baselines.get(m["base_key"], {})
        r2 = round2s.get(m["r2_key"],    {})
        r3 = round3s.get(m["r3_key"],    {})
        r4 = round4s.get(m["r4_key"],    {})
        r5 = round5s.get(m["r5_key"],    {})
        r6 = round6s.get(m["r6_key"],    {})

        acc_b.append(float(b.get("Top-1 Acc (%)", 0)))
        acc_r2.append(float(r2.get("Top-1 Acc (%)", 0)))
        acc_r3.append(float(r3.get("Top-1 Acc (%)", 0)))
        acc_r4.append(float(r4.get("Top-1 Acc (%)", 0)))
        acc_r5.append(float(r5.get("Top-1 Acc (%)", 0)))
        acc_r6.append(float(r6.get("Top-1 Acc (%)", 0)))

        em_b.append(float(b.get("Emissions (gCO\u2082)", 0)))
        em_r2.append(float(r2.get("Emissions (gCO\u2082)", 0)))
        em_r3.append(float(r3.get("Emissions (gCO\u2082)", 0)))
        em_r4.append(float(r4.get("Emissions (gCO\u2082)", 0)))
        em_r5.append(float(r5.get("Emissions (gCO\u2082)", 0)))
        em_r6.append(float(r6.get("Emissions (gCO\u2082)", 0)))

        time_b.append(float(b.get("Train Time (min)", 0)))
        time_r2.append(float(r2.get("Train Time (min)", 0)))
        time_r3.append(float(r3.get("Train Time (min)", 0)))
        time_r4.append(float(r4.get("Train Time (min)", 0)))
        time_r5.append(float(r5.get("Train Time (min)", 0)))
        time_r6.append(float(r6.get("Train Time (min)", 0)))

    x     = np.arange(len(models))
    width = 0.13

    # Palette (6 colors)
    c_base = "#94A3B8"   # slate    (Baseline)
    c_r2   = "#6366F1"   # indigo   (Round 2)
    c_r3   = "#10B981"   # emerald  (Round 3)
    c_r4   = "#F59E0B"   # amber    (Round 4)
    c_r5   = "#EC4899"   # rose     (Round 5)
    c_r6   = "#0EA5E9"   # sky blue (Round 6)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.suptitle(
        "GreenAI Experiment \u2014 6-Phase Comparison Dashboard",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Panel 1: Top-1 Accuracy ───────────────────────────────────────────────
    ax = axes[0]
    b0 = ax.bar(x - 2.5*width, acc_b,  width, label="Baseline (10 ep)",      color=c_base, edgecolor="none")
    b1 = ax.bar(x - 1.5*width, acc_r2, width, label="Round 2 (20 ep, reg)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x - 0.5*width, acc_r3, width, label="Round 3 (30 ep, reg)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 0.5*width, acc_r4, width, label="Round 4 (10 ep, reg)",  color=c_r4,   edgecolor="none")
    b4 = ax.bar(x + 1.5*width, acc_r5, width, label="Round 5 (20 ep, base)", color=c_r5,   edgecolor="none")
    b5 = ax.bar(x + 2.5*width, acc_r6, width, label="Round 6 (30 ep, base)", color=c_r6,   edgecolor="none")
    ax.set_title("Top-1 Accuracy (%)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(85, 98)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=7.5)
    for bars, vals, col in [
        (b0, acc_b,  "#475569"),
        (b1, acc_r2, "#3730A3"),
        (b2, acc_r3, "#065F46"),
        (b3, acc_r4, "#92400E"),
        (b4, acc_r5, "#9D174D"),
        (b5, acc_r6, "#0369A1"),
    ]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")

    # ── Panel 2: Carbon Emissions ─────────────────────────────────────────────
    ax = axes[1]
    b0 = ax.bar(x - 2.5*width, em_b,  width, label="Baseline (10 ep)",      color=c_base, edgecolor="none")
    b1 = ax.bar(x - 1.5*width, em_r2, width, label="Round 2 (20 ep, reg)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x - 0.5*width, em_r3, width, label="Round 3 (30 ep, reg)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 0.5*width, em_r4, width, label="Round 4 (10 ep, reg)",  color=c_r4,   edgecolor="none")
    b4 = ax.bar(x + 1.5*width, em_r5, width, label="Round 5 (20 ep, base)", color=c_r5,   edgecolor="none")
    b5 = ax.bar(x + 2.5*width, em_r6, width, label="Round 6 (30 ep, base)", color=c_r6,   edgecolor="none")
    ax.set_title("Carbon Emissions (gCO\u2082)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Emissions (gCO\u2082eq)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=7.5)
    for bars, vals, col in [
        (b0, em_b,  "#475569"),
        (b1, em_r2, "#3730A3"),
        (b2, em_r3, "#065F46"),
        (b3, em_r4, "#92400E"),
        (b4, em_r5, "#9D174D"),
        (b5, em_r6, "#0369A1"),
    ]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.25,
                        f"{v:.2f}g", ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")

    # ── Panel 3: Training Duration ────────────────────────────────────────────
    ax = axes[2]
    b0 = ax.bar(x - 2.5*width, time_b,  width, label="Baseline (10 ep)",      color=c_base, edgecolor="none")
    b1 = ax.bar(x - 1.5*width, time_r2, width, label="Round 2 (20 ep, reg)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x - 0.5*width, time_r3, width, label="Round 3 (30 ep, reg)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 0.5*width, time_r4, width, label="Round 4 (10 ep, reg)",  color=c_r4,   edgecolor="none")
    b4 = ax.bar(x + 1.5*width, time_r5, width, label="Round 5 (20 ep, base)", color=c_r5,   edgecolor="none")
    b5 = ax.bar(x + 2.5*width, time_r6, width, label="Round 6 (30 ep, base)", color=c_r6,   edgecolor="none")
    ax.set_title("Training Duration (min)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Duration (minutes)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=7.5)
    for bars, vals, col in [
        (b0, time_b,  "#475569"),
        (b1, time_r2, "#3730A3"),
        (b2, time_r3, "#065F46"),
        (b3, time_r4, "#92400E"),
        (b4, time_r5, "#9D174D"),
        (b5, time_r6, "#0369A1"),
    ]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                        f"{v:.1f}m", ha="center", va="bottom", fontsize=6, color=col, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PNG_DASH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved: {PNG_DASH}")


# ─── Markdown report ──────────────────────────────────────────────────────────

def make_markdown(baselines, round2s, round3s, round4s, round5s, round6s) -> None:
    """Generate final_report.md comparing all six phases."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build per-model data rows
    rows = []
    for m in MODEL_MAP:
        b  = baselines.get(m["base_key"], {})
        r2 = round2s.get(m["r2_key"],    {})
        r3 = round3s.get(m["r3_key"],    {})
        r4 = round4s.get(m["r4_key"],    {})
        r5 = round5s.get(m["r5_key"],    {})
        r6 = round6s.get(m["r6_key"],    {})
        rows.append({"m": m, "b": b, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6})

    # ── Accuracy table ────────────────────────────────────────────────────────
    acc_header = (
        "| Model | Baseline | Round 2 | \u0394 R2 | Round 3 | \u0394 R3 vs R2 | Round 4 | Round 5 | Round 6 | \u0394 R6 vs Base |\n"
        "|:------|:--------:|:-------:|:----:|:-------:|:----------:|:-------:|:-------:|:-------:|:------------:|"
    )
    acc_rows = ""
    for d in rows:
        b_acc  = d["b"].get("Top-1 Acc (%)", "N/A")
        r2_acc = d["r2"].get("Top-1 Acc (%)", "N/A")
        r3_acc = d["r3"].get("Top-1 Acc (%)", "N/A")
        r4_acc = d["r4"].get("Top-1 Acc (%)", "N/A")
        r5_acc = d["r5"].get("Top-1 Acc (%)", "N/A")
        r6_acc = d["r6"].get("Top-1 Acc (%)", "N/A")
        acc_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_acc, 2)}% "
            f"| {fv(r2_acc, 2)}% "
            f"| {delta(r2_acc, b_acc, 2)}% "
            f"| {fv(r3_acc, 2)}% "
            f"| {delta(r3_acc, r2_acc, 2)}% "
            f"| {fv(r4_acc, 2)}% "
            f"| {fv(r5_acc, 2)}% "
            f"| {fv(r6_acc, 2)}% "
            f"| {delta(r6_acc, b_acc, 2)}% |"
        )

    # ── Emissions table ───────────────────────────────────────────────────────
    em_header = (
        "| Model | Baseline | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | R6 vs Base |\n"
        "|:------|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|"
    )
    em_rows = ""
    for d in rows:
        b_em  = d["b"].get("Emissions (gCO\u2082)", "N/A")
        r2_em = d["r2"].get("Emissions (gCO\u2082)", "N/A")
        r3_em = d["r3"].get("Emissions (gCO\u2082)", "N/A")
        r4_em = d["r4"].get("Emissions (gCO\u2082)", "N/A")
        r5_em = d["r5"].get("Emissions (gCO\u2082)", "N/A")
        r6_em = d["r6"].get("Emissions (gCO\u2082)", "N/A")
        em_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_em, 3)}g "
            f"| {fv(r2_em, 3)}g "
            f"| {fv(r3_em, 3)}g "
            f"| {fv(r4_em, 3)}g "
            f"| {fv(r5_em, 3)}g "
            f"| {fv(r6_em, 3)}g "
            f"| **{pct_delta(r6_em, b_em)}** |"
        )

    # ── Energy table ──────────────────────────────────────────────────────────
    en_header = (
        "| Model | Baseline | Round 2 | Round 3 | Round 4 | Round 5 | Round 6 | R6 vs Base |\n"
        "|:------|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|"
    )
    en_rows = ""
    for d in rows:
        b_en  = d["b"].get("Energy (kWh)", "N/A")
        r2_en = d["r2"].get("Energy (kWh)", "N/A")
        r3_en = d["r3"].get("Energy (kWh)", "N/A")
        r4_en = d["r4"].get("Energy (kWh)", "N/A")
        r5_en = d["r5"].get("Energy (kWh)", "N/A")
        r6_en = d["r6"].get("Energy (kWh)", "N/A")
        en_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_en, 6)} "
            f"| {fv(r2_en, 6)} "
            f"| {fv(r3_en, 6)} "
            f"| {fv(r4_en, 6)} "
            f"| {fv(r5_en, 6)} "
            f"| {fv(r6_en, 6)} "
            f"| **{pct_delta(r6_en, b_en)}** |"
        )

    # ── Section 5: Baseline sweep analysis (R5 vs Base, R6 vs Base, R3 vs R6) ─
    r6_vs_base_rows = ""
    interpretations_r6_base = {
        "YOLO26": "Triplicar épocas baseline sem otimizações triplica o custo de emiss\u00f5es linearmente.",
        "CNN":    "30 \u00e9pocas sem regulariza\u00e7\u00e3o agravam o overfitting — R3 (otimizado) supera R6 com menos emiss\u00f5es.",
        "ViT":    "ViT sem hiperpar\u00e2metros refinados escala mal: 30 ep base gera alto custo ambiental."
    }
    for d in rows:
        model_id = d["m"]["id"]
        b_acc  = d["b"].get("Top-1 Acc (%)", "N/A")
        r6_acc = d["r6"].get("Top-1 Acc (%)", "N/A")
        b_em   = d["b"].get("Emissions (gCO\u2082)", "N/A")
        r6_em  = d["r6"].get("Emissions (gCO\u2082)", "N/A")
        interp = interpretations_r6_base.get(model_id, "")
        r6_vs_base_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {delta(r6_acc, b_acc, 2)}% "
            f"| {pct_delta(r6_em, b_em)} "
            f"| {interp} |"
        )

    r3_vs_r6_rows = ""
    interpretations_r3_r6 = {
        "YOLO26": "Round 3 (otimizado, 30 ep) vs Round 6 (base, 30 ep): mesmo or\u00e7amento de \u00e9pocas, diferentes hiperpar\u00e2metros.",
        "CNN":    "Regulariza\u00e7\u00e3o no R3 supera o baseline R6 mesmo com o mesmo n\u00famero de \u00e9pocas.",
        "ViT":    "LR e wd otimizados no R3 entregam superior accuracy ao R6 com baseline puro."
    }
    for d in rows:
        model_id = d["m"]["id"]
        r3_acc = d["r3"].get("Top-1 Acc (%)", "N/A")
        r6_acc = d["r6"].get("Top-1 Acc (%)", "N/A")
        r3_em  = d["r3"].get("Emissions (gCO\u2082)", "N/A")
        r6_em  = d["r6"].get("Emissions (gCO\u2082)", "N/A")
        interp = interpretations_r3_r6.get(model_id, "")
        r3_vs_r6_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {delta(r3_acc, r6_acc, 2)}% "
            f"| {pct_delta(r3_em, r6_em)} "
            f"| {interp} |"
        )

    # ── Hyperparameter detail sections ────────────────────────────────────────
    hp_sections = ""
    hp_map = {"YOLO26": "YOLO26", "CNN": "CNN", "ViT": "ViT"}
    label_map = {"YOLO26": "YOLO26", "CNN": "CNN", "ViT": "ViT (Transformers)"}
    for m in MODEL_MAP:
        hp = HYPERPARAMS[hp_map[m["id"]]]
        idx = [x["m"]["id"] for x in rows].index(m["id"])
        b  = rows[idx]["b"]
        r2 = rows[idx]["r2"]
        r3 = rows[idx]["r3"]
        r4 = rows[idx]["r4"]
        r5 = rows[idx]["r5"]
        r6 = rows[idx]["r6"]
        b_acc  = fv(b.get("Top-1 Acc (%)", "N/A"), 2)
        r2_acc = fv(r2.get("Top-1 Acc (%)", "N/A"), 2)
        r3_acc = fv(r3.get("Top-1 Acc (%)", "N/A"), 2)
        r4_acc = fv(r4.get("Top-1 Acc (%)", "N/A"), 2)
        r5_acc = fv(r5.get("Top-1 Acc (%)", "N/A"), 2)
        r6_acc = fv(r6.get("Top-1 Acc (%)", "N/A"), 2)
        hp_sections += f"""
### {label_map[m["id"]]}

| Phase | Hyperparameters |
|:------|:----------------|
| **Baseline** | {hp['baseline'].replace(chr(10), ' \u00b7 ')} |
| **Round 2** | {hp['round2'].replace(chr(10), ' \u00b7 ')} |
| **Round 3** | {hp['round3'].replace(chr(10), ' \u00b7 ')} |
| **Round 4** | {hp['round4'].replace(chr(10), ' \u00b7 ')} |
| **Round 5** | {hp['round5'].replace(chr(10), ' \u00b7 ')} |
| **Round 6** | {hp['round6'].replace(chr(10), ' \u00b7 ')} |

**Accuracy progression**: Base {b_acc}% \u2192 R2 {r2_acc}% \u2192 R3 {r3_acc}% \u2192 R4 {r4_acc}% \u2192 R5 {r5_acc}% \u2192 R6 {r6_acc}%

---
"""

    # ── Comparison Section: Baselines vs Modified ────────────────────────────
    comp_tables = ""
    budgets = [
        ("10 Épocas (Baseline Inicial vs Round 4)", "b", "r4"),
        ("20 Épocas (Round 5 vs Round 2)", "r5", "r2"),
        ("30 Épocas (Round 6 vs Round 3)", "r6", "r3"),
    ]
    for title, base_k, mod_k in budgets:
        comp_tables += f"### {title}\n\n"
        comp_tables += "| Model | Baseline Acc | Modificado Acc | Δ Acc | Baseline CO₂ | Modificado CO₂ | Δ CO₂ (%) | Baseline Tempo | Modificado Tempo | Δ Tempo (%) |\n"
        comp_tables += "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        for d in rows:
            base_row = d[base_k]
            mod_row = d[mod_k]
            b_acc = base_row.get("Top-1 Acc (%)", "N/A")
            m_acc = mod_row.get("Top-1 Acc (%)", "N/A")
            b_em = base_row.get("Emissions (gCO₂)", "N/A")
            m_em = mod_row.get("Emissions (gCO₂)", "N/A")
            b_time = base_row.get("Train Time (min)", "N/A")
            m_time = mod_row.get("Train Time (min)", "N/A")
            
            comp_tables += (
                f"| **{d['m']['label']}** "
                f"| {fv(b_acc, 2)}% "
                f"| {fv(m_acc, 2)}% "
                f"| **{delta(m_acc, b_acc, 2)}%** "
                f"| {fv(b_em, 3)}g "
                f"| {fv(m_em, 3)}g "
                f"| **{pct_delta(m_em, b_em)}** "
                f"| {fv(b_time, 1)} min "
                f"| {fv(m_time, 1)} min "
                f"| **{pct_delta(m_time, b_time)}** |\n"
            )
        comp_tables += "\n"

    md = f"""# GreenAI Experiment \u2014 Final Report (6-Phase Comparison)

Generated: {ts}

This report summarises the complete six-phase experimental study comparing CNN,
YOLO26, and ViT (DeiT-Tiny) models trained on Fashion-MNIST, with successive
hyperparameter refinements aimed at improving accuracy while monitoring
energy consumption and carbon emissions.

---

## 1. Experimental Phases Overview

| Phase | Epochs | Batch | Key Goal |
|:------|:------:|:-----:|:---------|
| **Baseline** | 10 | 16 | Establish reference performance with default hyperparameters |
| **Round 2 (Modified)** | 20 | 32 | Address overfitting/underfitting; add regularization & augmentation |
| **Round 3** | 30 | 32 | Fine-tune further; more epochs; balance LR decay and regularization |
| **Round 4 (Efficiency)** | 10 | 32 | Round 3 hyperparams at Baseline epoch budget \u2014 isolate hyperparameter effect |
| **Round 5 (20-ep Baseline)** | 20 | 16 | Baseline hyperparams at Round 2 epoch budget \u2014 isolate hyperparameter improvement at 20 epochs |
| **Round 6 (30-ep Baseline)** | 30 | 16 | Baseline hyperparams at Round 3 epoch budget \u2014 final ceiling of pure baseline scaling |

---

## 2. Final Accuracy Comparison (Top-1, Test Set)

{acc_header}{acc_rows}

> \u0394 values are absolute percentage point differences.

---

## 3. Carbon Emissions Comparison (gCO\u2082eq)

{em_header}{em_rows}

> All values in grams of CO\u2082 equivalent (gCO\u2082eq) measured by [CodeCarbon](https://codecarbon.io/).

---

## 4. Energy Consumption Comparison (kWh)

{en_header}{en_rows}

---

## 5. Round 6 \u2014 Efficiency and Hyperparameter Analysis

### Round 6 vs Baseline (same hyperparams, different epochs: 30 vs 10)

| Model | \u0394 Accuracy | \u0394 CO\u2082 | Interpretation |
|:------|:----------:|:-----:|:---------------|{r6_vs_base_rows}

### Round 3 vs Round 6 (same epochs: 30, modified vs baseline hyperparams)

| Model | \u0394 Accuracy | \u0394 CO\u2082 | Interpretation |
|:------|:----------:|:-----:|:---------------|{r3_vs_r6_rows}

---

## 6. Comparação Direta Detalhada: Baselines vs Modificados

{comp_tables}

---

## 7. Final Dashboard Visualization

![Final Dashboard](final_dashboard.png)

---

## 8. Hyperparameter Evolution by Model

{hp_sections}

## 9. Key Takeaways (Baselines vs Modificados)

- **Aumentar Épocas Sem Regularização é Ineficiente**: O escalamento dos baselines puros (Baseline 10ep → R5 20ep → R6 30ep) resultou em perdas de acurácia (overfitting) para YOLO26 (-0.43%) e CNN (-0.20%), enquanto as emissões de carbono explodiram (+284.6% no YOLO26 e +1041.0% na CNN). O ViT teve ganho irrisório de +0.04% ao custo de +129.3% de CO₂.
- **Otimizações GreenAI Garantem Escalabilidade**: Ao utilizar regularização (dropout, label smoothing) e aumentações de dados nos modificados (R4 10ep → R2 20ep → R3 30ep), os modelos mantiveram a capacidade de generalização e escalaram com eficiência, entregando ganhos significativos (YOLO26 obteve +1.11% de ganho de acurácia líquida).
- **YOLO26 com AdamW e Batch 32 é Altamente Eficiente**: Em todos os orçamentos de épocas (10, 20 e 30), a versão modificada do YOLO26 superou amplamente a versão baseline, reduzindo o tempo de treino em até **61.9%** e emissões de carbono em até **60.2%**, com acurácia superior.
- **CNN & Regularização Exigem Épocas**: A CNN modificada precisa de mais tempo de convergência (30 épocas) para compensar a regularização e o aumento de dados geométricos. O processamento das augmentações em tempo real aumenta a carga de CPU, elevando o consumo energético, mas o resultado final em generalização e acurácia (+0.72% vs R6) compensa o custo ambiental para longos treinamentos.

---

*Report generated automatically by `generate_final_report.py`.*
"""

    MD_REPORT.write_text(md, encoding="utf-8")
    print(f"Final report saved: {MD_REPORT}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  Generating GreenAI Final Report (6-Phase Comparison)")
    print("=" * 65)

    baselines = read_csv(CSV_BASELINE, "Baseline")
    round2s   = read_csv(CSV_ROUND2,   "Model")
    round3s   = read_csv(CSV_ROUND3,   "Model")
    round4s   = read_csv(CSV_ROUND4,   "Model")
    round5s   = read_csv(CSV_ROUND5,   "Model")
    round6s   = read_csv(CSV_ROUND6,   "Model")

    missing = []
    if not baselines: missing.append(str(CSV_BASELINE))
    if not round2s:   missing.append(str(CSV_ROUND2))
    if not round3s:   missing.append(str(CSV_ROUND3))
    if not round4s:   missing.append(str(CSV_ROUND4))
    if not round5s:   missing.append(str(CSV_ROUND5))
    if not round6s:   missing.append(str(CSV_ROUND6))

    if missing:
        print("\n[WARNING] Missing input files:")
        for f in missing:
            print(f"  \u2717 {f}")
        print("\nNote: Some rounds have not been run or are missing. A partial report will be generated.")

    make_dashboard(baselines, round2s, round3s, round4s, round5s, round6s)
    make_markdown(baselines, round2s, round3s, round4s, round5s, round6s)

    print(f"\n\u2713 All outputs written to: {REPORTS_DIR}")
    print(f"  \u2022 {MD_REPORT.name}")
    print(f"  \u2022 {PNG_DASH.name}")


if __name__ == "__main__":
    main()
