"""
report.py — Summary report generator for the GreenAI experiment.

Reads per-baseline artifacts produced by run.py:
  • <baseline>/output/runs/train/results.csv  — training metrics (last epoch)
  • <baseline>/output/emissions.csv           — CodeCarbon energy/emissions data

Outputs (written to reports/):
  • summary_metrics.csv    — machine-readable table (metrics + emissions)
  • summary_report.md      — human-readable Markdown report

Usage (standalone):
    python report.py

Called automatically by main.py after all baselines finish.
"""

import csv
from datetime import datetime
from pathlib import Path

# ─── Baseline registry ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

BASELINES = {
    "YOLO26": {
        "results_csv":   ROOT / "yolo26_baseline"      / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "yolo26_baseline"      / "output" / "emissions.csv",
    },
    "CNN": {
        "results_csv":   ROOT / "cnn_baseline"         / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "cnn_baseline"         / "output" / "emissions.csv",
    },
    "ViT (Transformers)": {
        "results_csv":   ROOT / "transformers_baseline" / "output" / "runs" / "train" / "results.csv",
        "emissions_csv": ROOT / "transformers_baseline" / "output" / "emissions.csv",
    },
}

REPORTS_DIR = ROOT / "reports"


# ─── Readers ─────────────────────────────────────────────────────────────────

def _read_last_row(path: Path) -> dict | None:
    """Return the last data row of a CSV as a dict, or None if missing/empty."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _read_emissions(path: Path) -> dict | None:
    """
    Return the last row of a CodeCarbon emissions.csv as a dict.
    CodeCarbon appends one row per tracker.stop() call.
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
    """Collect metrics and emissions for every baseline."""
    rows = []

    for name, paths in BASELINES.items():
        metrics  = _read_last_row(paths["results_csv"])
        emissions = _read_emissions(paths["emissions_csv"])

        row = {"Baseline": name}

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
    with open(path, "w", newline="") as f:
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
    metric_cols   = ["Baseline", "Epochs", "Train Loss", "Val Loss",
                     "Top-1 Acc (%)", "Top-5 Acc (%)", "Train Time (min)"]
    emission_cols = ["Baseline", "Duration (s)", "Energy (kWh)",
                     "GPU Energy (kWh)", "CPU Energy (kWh)", "RAM Energy (kWh)",
                     "Emissions (gCO₂)"]

    metric_rows   = [{c: r[c] for c in metric_cols}   for r in rows]
    emission_rows = [{c: r[c] for c in emission_cols} for r in rows]

    md = f"""# GreenAI Experiment — Summary Report

Generated: {ts}

## Experiment Setup

| Parameter | Value |
|---|---|
| Dataset | Fashion-MNIST (10 classes, 60 k train / 10 k test) |
| Epochs | 10 |
| Batch size | 16 |
| Seed | 0 |
| Precision | AMP (FP16 where available) |

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

| Baseline | Results CSV | Emissions CSV |
|---|---|---|
"""
    for name, paths in BASELINES.items():
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
    print("  Generating experiment summary report")
    print("=" * 60)

    rows = collect_results()

    # Print to console
    print(f"\n{'Baseline':<22} {'Top-1 Acc':>10} {'Val Loss':>10} "
          f"{'Energy (kWh)':>14} {'Emissions (gCO₂)':>17}")
    print("-" * 76)
    for r in rows:
        print(f"{r['Baseline']:<22} {r['Top-1 Acc (%)']:>10} {r['Val Loss']:>10} "
              f"{r['Energy (kWh)']:>14} {r['Emissions (gCO₂)']:>17}")

    write_csv(rows,      REPORTS_DIR / "summary_metrics.csv")
    write_markdown(rows, REPORTS_DIR / "summary_report.md")

    print(f"\nReport saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
