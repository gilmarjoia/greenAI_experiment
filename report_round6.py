"""
report_round6.py — Summary report generator for the GreenAI Round 6 experiment.

Reads per-model artifacts produced by run.py:
  • <model>_round6/output/runs/train/results.csv  — training metrics (last epoch)
  • <model>_round6/output/emissions.csv           — CodeCarbon energy/emissions data

Outputs (written to reports/):
  • summary_metrics_round6.csv    — machine-readable table (metrics + emissions)
  • summary_report_round6.md      — human-readable Markdown report

Usage (standalone):
    python report_round6.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 for console output to handle special characters like gCO₂
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── Model registry ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

ROUND6_MODELS = {
    "YOLO26 Round 6": {
        "results_csv":   ROOT / "yolo26_round6"       / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "yolo26_round6"       / "output" / "emissions.csv",
    },
    "CNN Round 6": {
        "results_csv":   ROOT / "cnn_round6"          / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "cnn_round6"          / "output" / "emissions.csv",
    },
    "ViT (Transformers) Round 6": {
        "results_csv":   ROOT / "transformers_round6"  / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "transformers_round6"  / "output" / "emissions.csv",
    },
}

REPORTS_DIR = ROOT / "reports"


# ─── Readers ─────────────────────────────────────────────────────────────────

def _read_last_row(path: Path) -> dict | None:
    """Return the last data row of a CSV as a dict, or None if missing/empty."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _read_emissions(path: Path) -> dict | None:
    """
    Return the last row of a CodeCarbon emissions.csv as a dict.
    """
    return _read_last_row(path)


def _safe_float(value, decimals: int = 4) -> str:
    """Convert to float rounded to `decimals`, or return 'N/A'."""
    try:
        return str(round(float(value), decimals))
    except (TypeError, ValueError):
        return "N/A"


# ─── Data collection ─────────────────────────────────────────────────────────

def collect_results() -> list[dict]:
    """Collect metrics and emissions for every Round 6 model."""
    rows = []

    for name, paths in ROUND6_MODELS.items():
        metrics   = _read_last_row(paths["results_csv"])
        emissions = _read_emissions(paths["emissions_csv"])

        row = {"Model": name}

        # ── Training metrics (last epoch) ─────────────────────────────────────
        if metrics:
            row["Epochs"]           = _safe_float(metrics.get("epoch"),                     0)
            row["Train Loss"]       = _safe_float(metrics.get("train/loss"),                 5)
            row["Val Loss"]         = _safe_float(metrics.get("val/loss"),                   5)
            row["Top-1 Acc (%)"]    = _safe_float(
                float(metrics.get("metrics/accuracy_top1", "nan")) * 100, 2
            )
            row["Top-5 Acc (%)"]    = _safe_float(
                float(metrics.get("metrics/accuracy_top5", "nan")) * 100, 2
            )
            row["Train Time (min)"] = _safe_float(
                float(metrics.get("time", "nan")) / 60, 2
            )
        else:
            for col in ["Epochs", "Train Loss", "Val Loss",
                        "Top-1 Acc (%)", "Top-5 Acc (%)", "Train Time (min)"]:
                row[col] = "N/A"

        # ── Emissions (CodeCarbon) ─────────────────────────────────────────────
        if emissions:
            row["Energy (kWh)"]     = _safe_float(emissions.get("energy_consumed"),         6)
            row["Emissions (gCO₂)"] = _safe_float(
                float(emissions.get("emissions", "nan")) * 1000, 4   # kg → g
            )
            row["GPU Energy (kWh)"] = _safe_float(emissions.get("gpu_energy"),              6)
            row["CPU Energy (kWh)"] = _safe_float(emissions.get("cpu_energy"),              6)
            row["RAM Energy (kWh)"] = _safe_float(emissions.get("ram_energy"),              6)
            row["Duration (s)"]     = _safe_float(emissions.get("duration"),                2)
        else:
            for col in ["Energy (kWh)", "Emissions (gCO₂)", "GPU Energy (kWh)",
                        "CPU Energy (kWh)", "RAM Energy (kWh)", "Duration (s)"]:
                row[col] = "N/A"

        rows.append(row)

    return rows


# ─── Writers ─────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    """Write results to a machine-readable CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def _md_table(rows: list[dict]) -> str:
    """Render a list of dicts as a Markdown table."""
    if not rows:
        return "_No data available._\n"
    headers = list(rows[0].keys())
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)    + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write a human-readable Markdown summary report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Split into two tables for readability
    metric_cols   = ["Model", "Epochs", "Train Loss", "Val Loss",
                     "Top-1 Acc (%)", "Top-5 Acc (%)", "Train Time (min)"]
    emission_cols = ["Model", "Duration (s)", "Energy (kWh)",
                     "GPU Energy (kWh)", "CPU Energy (kWh)", "RAM Energy (kWh)",
                     "Emissions (gCO₂)"]

    metric_rows   = [{c: r[c] for c in metric_cols}   for r in rows]
    emission_rows = [{c: r[c] for c in emission_cols} for r in rows]

    md = f"""# GreenAI Experiment — Round 6 Summary Report

Generated: {ts}

## Experiment Setup (Round 6 — 30 Epoch Baseline — Final Round)

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 30 |
| Batch size | 16 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

> **Goal**: Evaluate baseline hyperparameters (no data augmentations or extra
> regularization beyond baseline defaults) with triple the baseline epochs (30).
> This is the **final** experiment round, establishing the 30-epoch baseline ceiling.

---

## Training Metrics (last epoch)

{_md_table(metric_rows)}
> **Top-1 / Top-5 accuracy** measured on the test split.
> **Train Time** is cumulative wall-clock time at the last epoch.

---

## Energy Consumption & Carbon Emissions

{_md_table(emission_rows)}
> **Energy** and **Emissions** cover the full training run (load dataset + train model tasks).
> Emissions reported in grams of CO₂ equivalent (gCO₂eq).
> Source: [CodeCarbon](https://codecarbon.io/)

---

## Files

| Model | Results CSV | Emissions CSV |
|---|---|---|
"""
    for name, paths in ROUND6_MODELS.items():
        r = paths["results_csv"]
        e = paths["emissions_csv"]
        r_str = "✓" if r.exists() else "✗ missing"
        e_str = "✓" if e.exists() else "✗ missing"
        md += f"| {name} | {r_str} | {e_str} |\n"

    path.write_text(md, encoding="utf-8")
    print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Generating experiment summary report for Round 6 models")
    print("=" * 60)

    rows = collect_results()

    # Print to console
    print(f"\n{'Model':<30} {'Top-1 Acc':>10} {'Val Loss':>10} "
          f"{'Energy (kWh)':>14} {'Emissions (gCO₂)':>17}")
    print("-" * 84)
    for r in rows:
        print(f"{r['Model']:<30} {r['Top-1 Acc (%)']:>10} {r['Val Loss']:>10} "
              f"{r['Energy (kWh)']:>14} {r['Emissions (gCO₂)']:>17}")

    write_csv(rows,      REPORTS_DIR / "summary_metrics_round6.csv")
    write_markdown(rows, REPORTS_DIR / "summary_report_round6.md")

    print(f"\nReport saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
