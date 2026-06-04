"""
generate_final_report.py — Final consolidated comparison report generator.

Reads summary metrics from all four experimental phases:
  - Baseline      (reports/summary_metrics.csv)
  - Modified/R2   (reports/summary_metrics_modified.csv)
  - Round 3       (reports/summary_metrics_round3.csv)
  - Round 4       (reports/summary_metrics_round4.csv)

Produces:
  - reports/final_report.md          — Full Markdown report (4-phase comparison)
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
MD_REPORT    = REPORTS_DIR / "final_report.md"
PNG_DASH     = REPORTS_DIR / "final_dashboard.png"

# ─── Hyperparameter tables (all 4 rounds) ─────────────────────────────────────
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
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_csv(path: Path, key_col: str) -> dict:
    """Read CSV into dict keyed by model name."""
    data = {}
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
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
        "label":    "YOLO26",
    },
    {
        "id":       "CNN",
        "base_key": "CNN",
        "r2_key":   "CNN Modified",
        "r3_key":   "CNN Round 3",
        "r4_key":   "CNN Round 4",
        "label":    "CNN",
    },
    {
        "id":       "ViT",
        "base_key": "ViT (Transformers)",
        "r2_key":   "ViT (Transformers) Modified",
        "r3_key":   "ViT (Transformers) Round 3",
        "r4_key":   "ViT (Transformers) Round 4",
        "label":    "ViT",
    },
]


# ─── Plot ─────────────────────────────────────────────────────────────────────

def make_dashboard(baselines, round2s, round3s, round4s) -> None:
    """Generate a 1x3 comparison dashboard PNG with 4 bars per model."""
    models = [m["label"] for m in MODEL_MAP]

    acc_b, acc_r2, acc_r3, acc_r4       = [], [], [], []
    em_b,  em_r2,  em_r3,  em_r4        = [], [], [], []
    time_b, time_r2, time_r3, time_r4    = [], [], [], []

    for m in MODEL_MAP:
        b  = baselines.get(m["base_key"], {})
        r2 = round2s.get(m["r2_key"],    {})
        r3 = round3s.get(m["r3_key"],    {})
        r4 = round4s.get(m["r4_key"],    {})

        acc_b.append(float(b.get("Top-1 Acc (%)", 0)))
        acc_r2.append(float(r2.get("Top-1 Acc (%)", 0)))
        acc_r3.append(float(r3.get("Top-1 Acc (%)", 0)))
        acc_r4.append(float(r4.get("Top-1 Acc (%)", 0)))

        em_b.append(float(b.get("Emissions (gCO₂)", 0)))
        em_r2.append(float(r2.get("Emissions (gCO₂)", 0)))
        em_r3.append(float(r3.get("Emissions (gCO₂)", 0)))
        em_r4.append(float(r4.get("Emissions (gCO₂)", 0)))

        time_b.append(float(b.get("Train Time (min)", 0)))
        time_r2.append(float(r2.get("Train Time (min)", 0)))
        time_r3.append(float(r3.get("Train Time (min)", 0)))
        time_r4.append(float(r4.get("Train Time (min)", 0)))

    x     = np.arange(len(models))
    width = 0.18

    # Palette
    c_base = "#94A3B8"   # slate (Baseline)
    c_r2   = "#6366F1"   # indigo (Round 2)
    c_r3   = "#10B981"   # emerald (Round 3)
    c_r4   = "#F59E0B"   # amber (Round 4)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "GreenAI Experiment — 4-Phase Comparison Dashboard",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Panel 1: Top-1 Accuracy ───────────────────────────────────────────────
    ax = axes[0]
    b0 = ax.bar(x - 1.5*width, acc_b,  width, label="Baseline (10 ep)", color=c_base, edgecolor="none")
    b1 = ax.bar(x - 0.5*width, acc_r2, width, label="Round 2 (20 ep)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x + 0.5*width, acc_r3, width, label="Round 3 (30 ep)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 1.5*width, acc_r4, width, label="Round 4 (10 ep)",  color=c_r4,   edgecolor="none")
    ax.set_title("Top-1 Accuracy (%)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(85, 98)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    for bars, vals, col in [
        (b0, acc_b, "#475569"), 
        (b1, acc_r2, "#3730A3"), 
        (b2, acc_r3, "#065F46"),
        (b3, acc_r4, "#92400E")
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

    # ── Panel 2: Carbon Emissions ─────────────────────────────────────────────
    ax = axes[1]
    b0 = ax.bar(x - 1.5*width, em_b,  width, label="Baseline (10 ep)", color=c_base, edgecolor="none")
    b1 = ax.bar(x - 0.5*width, em_r2, width, label="Round 2 (20 ep)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x + 0.5*width, em_r3, width, label="Round 3 (30 ep)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 1.5*width, em_r4, width, label="Round 4 (10 ep)",  color=c_r4,   edgecolor="none")
    ax.set_title("Carbon Emissions (gCO₂)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Emissions (gCO₂eq)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    for bars, vals, col in [
        (b0, em_b, "#475569"), 
        (b1, em_r2, "#3730A3"), 
        (b2, em_r3, "#065F46"),
        (b3, em_r4, "#92400E")
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.25,
                    f"{v:.2f}g", ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

    # ── Panel 3: Training Duration ────────────────────────────────────────────
    ax = axes[2]
    b0 = ax.bar(x - 1.5*width, time_b,  width, label="Baseline (10 ep)", color=c_base, edgecolor="none")
    b1 = ax.bar(x - 0.5*width, time_r2, width, label="Round 2 (20 ep)",  color=c_r2,   edgecolor="none")
    b2 = ax.bar(x + 0.5*width, time_r3, width, label="Round 3 (30 ep)",  color=c_r3,   edgecolor="none")
    b3 = ax.bar(x + 1.5*width, time_r4, width, label="Round 4 (10 ep)",  color=c_r4,   edgecolor="none")
    ax.set_title("Training Duration (min)", fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Duration (minutes)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8)
    for bars, vals, col in [
        (b0, time_b, "#475569"), 
        (b1, time_r2, "#3730A3"), 
        (b2, time_r3, "#065F46"),
        (b3, time_r4, "#92400E")
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                    f"{v:.1f}m", ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PNG_DASH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard saved: {PNG_DASH}")


# ─── Markdown report ──────────────────────────────────────────────────────────

def make_markdown(baselines, round2s, round3s, round4s) -> None:
    """Generate final_report.md comparing all four phases."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build per-model data rows
    rows = []
    for m in MODEL_MAP:
        b  = baselines.get(m["base_key"], {})
        r2 = round2s.get(m["r2_key"],    {})
        r3 = round3s.get(m["r3_key"],    {})
        r4 = round4s.get(m["r4_key"],    {})
        rows.append({"m": m, "b": b, "r2": r2, "r3": r3, "r4": r4})

    # ── Accuracy table ────────────────────────────────────────────────────────
    acc_header = (
        "| Model | Baseline | Round 2 | Δ R2 | Round 3 | Δ R3 vs R2 | Round 4 | Δ R4 vs Base |\n"
        "|:------|:--------:|:-------:|:----:|:-------:|:----------:|:-------:|:------------:|"
    )
    acc_rows = ""
    for d in rows:
        b_acc  = d["b"].get("Top-1 Acc (%)", "N/A")
        r2_acc = d["r2"].get("Top-1 Acc (%)", "N/A")
        r3_acc = d["r3"].get("Top-1 Acc (%)", "N/A")
        r4_acc = d["r4"].get("Top-1 Acc (%)", "N/A")
        acc_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_acc, 2)}% "
            f"| {fv(r2_acc, 2)}% "
            f"| {delta(r2_acc, b_acc, 2)}% "
            f"| {fv(r3_acc, 2)}% "
            f"| {delta(r3_acc, r2_acc, 2)}% "
            f"| {fv(r4_acc, 2)}% "
            f"| {delta(r4_acc, b_acc, 2)}% |"
        )

    # ── Emissions table ───────────────────────────────────────────────────────
    em_header = (
        "| Model | Baseline | Round 2 | Round 3 | Round 4 | R4 vs Base |\n"
        "|:------|:--------:|:-------:|:-------:|:-------:|:----------:|"
    )
    em_rows = ""
    for d in rows:
        b_em  = d["b"].get("Emissions (gCO₂)", "N/A")
        r2_em = d["r2"].get("Emissions (gCO₂)", "N/A")
        r3_em = d["r3"].get("Emissions (gCO₂)", "N/A")
        r4_em = d["r4"].get("Emissions (gCO₂)", "N/A")
        em_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_em, 3)}g "
            f"| {fv(r2_em, 3)}g "
            f"| {fv(r3_em, 3)}g "
            f"| {fv(r4_em, 3)}g "
            f"| **{pct_delta(r4_em, b_em)}** |"
        )

    # ── Energy table ──────────────────────────────────────────────────────────
    en_header = (
        "| Model | Baseline | Round 2 | Round 3 | Round 4 | R4 vs Base |\n"
        "|:------|:--------:|:-------:|:-------:|:-------:|:----------:|"
    )
    en_rows = ""
    for d in rows:
        b_en  = d["b"].get("Energy (kWh)", "N/A")
        r2_en = d["r2"].get("Energy (kWh)", "N/A")
        r3_en = d["r3"].get("Energy (kWh)", "N/A")
        r4_en = d["r4"].get("Energy (kWh)", "N/A")
        en_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {fv(b_en, 6)} "
            f"| {fv(r2_en, 6)} "
            f"| {fv(r3_en, 6)} "
            f"| {fv(r4_en, 6)} "
            f"| **{pct_delta(r4_en, b_en)}** |"
        )

    # ── Section 5: Round 4 vs Baseline ────────────────────────────────────────
    r4_vs_base_rows = ""
    interpretations_base = {
        "YOLO26": "Hiperparâmetros melhorados oferecem mesma acurácia com 60% menos emissões",
        "CNN": "Regularização aumenta custo; ganho de acurácia exige mais épocas para maturar",
        "ViT": "Hiperparâmetros refinados já superam o Baseline com mesmo orçamento de épocas"
    }
    for d in rows:
        model_id = d["m"]["id"]
        b_acc  = d["b"].get("Top-1 Acc (%)", "N/A")
        r4_acc = d["r4"].get("Top-1 Acc (%)", "N/A")
        b_em   = d["b"].get("Emissions (gCO₂)", "N/A")
        r4_em  = d["r4"].get("Emissions (gCO₂)", "N/A")
        interp = interpretations_base.get(model_id, "")
        r4_vs_base_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {delta(r4_acc, b_acc, 2)}% "
            f"| {pct_delta(r4_em, b_em)} "
            f"| {interp} |"
        )

    # ── Section 5: Round 4 vs Round 3 ────────────────────────────────────────
    r4_vs_r3_rows = ""
    interpretations_r3 = {
        "YOLO26": "Os 20 épocas extras do Round 3 foram cruciais para o ganho de acurácia",
        "CNN": "Regularização precisa de mais épocas para convergir plenamente",
        "ViT": "Hiperparâmetros são o principal driver; épocas extras contribuem marginalmente"
    }
    for d in rows:
        model_id = d["m"]["id"]
        r3_acc = d["r3"].get("Top-1 Acc (%)", "N/A")
        r4_acc = d["r4"].get("Top-1 Acc (%)", "N/A")
        r3_em  = d["r3"].get("Emissions (gCO₂)", "N/A")
        r4_em  = d["r4"].get("Emissions (gCO₂)", "N/A")
        interp = interpretations_r3.get(model_id, "")
        if model_id == "YOLO26":
            d_acc = float(r3_acc) - float(r4_acc)
            interp = f"Os 20 épocas extras do Round 3 foram cruciais para o ganho de +{d_acc:.2f}%"
        r4_vs_r3_rows += (
            f"\n| **{d['m']['label']}** "
            f"| {delta(r4_acc, r3_acc, 2)}% "
            f"| {pct_delta(r4_em, r3_em)} "
            f"| {interp} |"
        )

    # ── Hyperparameter detail sections ────────────────────────────────────────
    hp_sections = ""
    hp_map = {"YOLO26": "YOLO26", "CNN": "CNN", "ViT": "ViT"}
    label_map = {"YOLO26": "YOLO26", "CNN": "CNN", "ViT": "ViT (Transformers)"}
    for m in MODEL_MAP:
        hp = HYPERPARAMS[hp_map[m["id"]]]
        b  = rows[[x["m"]["id"] for x in rows].index(m["id"])]["b"]
        r2 = rows[[x["m"]["id"] for x in rows].index(m["id"])]["r2"]
        r3 = rows[[x["m"]["id"] for x in rows].index(m["id"])]["r3"]
        r4 = rows[[x["m"]["id"] for x in rows].index(m["id"])]["r4"]
        b_acc  = fv(b.get("Top-1 Acc (%)", "N/A"), 2)
        r2_acc = fv(r2.get("Top-1 Acc (%)", "N/A"), 2)
        r3_acc = fv(r3.get("Top-1 Acc (%)", "N/A"), 2)
        r4_acc = fv(r4.get("Top-1 Acc (%)", "N/A"), 2)
        hp_sections += f"""
### {label_map[m["id"]]}

| Phase | Hyperparameters |
|:------|:----------------|
| **Baseline** | {hp['baseline'].replace(chr(10), ' · ')} |
| **Round 2** | {hp['round2'].replace(chr(10), ' · ')} |
| **Round 3** | {hp['round3'].replace(chr(10), ' · ')} |
| **Round 4** | {hp['round4'].replace(chr(10), ' · ')} |

**Accuracy progression**: Baseline {b_acc}% → Round 2 {r2_acc}% → Round 3 {r3_acc}% → Round 4 {r4_acc}%

---
"""

    md = f"""# GreenAI Experiment — Final Report (4-Phase Comparison)

Generated: {ts}

This report summarises the complete four-phase experimental study comparing CNN,
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
| **Round 4 (Efficiency)** | 10 | 32 | Round 3 hyperparams at Baseline epoch budget — isolate hyperparameter effect |

---

## 2. Final Accuracy Comparison (Top-1, Test Set)

{acc_header}{acc_rows}

> Δ values are absolute percentage point differences.

---

## 3. Carbon Emissions Comparison (gCO₂eq)

{em_header}{em_rows}

> All values in grams of CO₂ equivalent (gCO₂eq) measured by [CodeCarbon](https://codecarbon.io/).

---

## 4. Energy Consumption Comparison (kWh)

{en_header}{en_rows}

---

## 5. Round 4 — Efficiency Benchmark Analysis

### Round 4 vs Baseline (same epochs, different hyperparams)

| Model | Δ Accuracy | Δ CO₂ | Interpretation |
|:------|:----------:|:-----:|:---------------|{r4_vs_base_rows}

### Round 4 vs Round 3 (same hyperparams, different epochs)

| Model | Δ Accuracy | CO₂ saved | Interpretation |
|:------|:----------:|:---------:|:---------------|{r4_vs_r3_rows}

---

## 6. Final Dashboard Visualization

![Final Dashboard](final_dashboard.png)

---

## 7. Hyperparameter Evolution by Model

{hp_sections}

## 8. Key Takeaways

- **ViT** é o mais beneficiado pelos hiperparâmetros refinados: com apenas 10 épocas (Round 4), supera o Baseline em +0.11% *e* consome 16% menos energia. O ajuste de LR, label smoothing e augmentation são o principal driver de performance.

- **YOLO26** com hiperparâmetros otimizados (Round 4) mantém praticamente a mesma acurácia do Baseline (+0.07%) usando **60% menos energia** — evidência forte de que o AdamW e batch maior tornam o treinamento mais eficiente energeticamente.

- **CNN** precisa de mais épocas para que a regularização (dropout + augmentation) maturize. No Round 4, com 10 épocas, os hiperparâmetros melhorados quase não agregam acurácia vs Baseline (-0.13%), mas o Round 3 com 30 épocas mostra o ganho real (+0.52%).

- **Melhor trade-off acurácia × sustentabilidade**: **Round 4 YOLO26** (90.63% de acurácia com apenas 0.768g CO₂) é a opção mais verde do experimento inteiro.

- **Melhor acurácia absoluta**: **Round 3 ViT** com 95.18%.

- **GreenAI Insight**: Hiperparâmetros bem ajustados podem reduzir o custo energético do treinamento **sem sacrificar acurácia** (ViT Round 4 vs Baseline), mas para modelos menores como CNN e YOLO26, mais épocas continuam sendo necessárias para extrair o potencial completo da regularização.

---

*Report generated automatically by `generate_final_report.py`.*
"""

    MD_REPORT.write_text(md, encoding="utf-8")
    print(f"Final report saved: {MD_REPORT}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  Generating GreenAI Final Report (4-Phase Comparison)")
    print("=" * 65)

    baselines = read_csv(CSV_BASELINE, "Baseline")
    round2s   = read_csv(CSV_ROUND2,   "Model")
    round3s   = read_csv(CSV_ROUND3,   "Model")
    round4s   = read_csv(CSV_ROUND4,   "Model")

    missing = []
    if not baselines: missing.append(str(CSV_BASELINE))
    if not round2s:   missing.append(str(CSV_ROUND2))
    if not round3s:   missing.append(str(CSV_ROUND3))
    if not round4s:   missing.append(str(CSV_ROUND4))

    if missing:
        print("\n[ERROR] Missing input files:")
        for f in missing:
            print(f"  ✗ {f}")
        print("\nRun the individual report scripts first:")
        print("  python report.py")
        print("  python report_modified.py")
        print("  python report_round3.py")
        print("  python report_round4.py")
        sys.exit(1)

    make_dashboard(baselines, round2s, round3s, round4s)
    make_markdown(baselines, round2s, round3s, round4s)

    print(f"\n✓ All outputs written to: {REPORTS_DIR}")
    print(f"  • {MD_REPORT.name}")
    print(f"  • {PNG_DASH.name}")


if __name__ == "__main__":
    main()
