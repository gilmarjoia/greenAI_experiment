"""
main_round6.py — Orchestrator for the GreenAI Round 6 experiment phase.

Runs the three Round 6 models in sequence:
    1. YOLO26  (yolo26_round6/run.py)
    2. CNN     (cnn_round6/run.py)
    3. ViT     (transformers_round6/run.py)

Goal of Round 6 — 30 Epoch Baseline:
  Each model uses the baseline hyperparameters (without regularization or augmentation
  improvements) but trains for 30 epochs. This is the final experiment round.

Each Round 6 run:
  - Manages its own CodeCarbon tracker internally
  - Writes artifacts under  <model>_round6/output/runs/train/
  - Saves an emissions.csv   under  <model>_round6/output/

After all runs, report_round6.py is called to generate a summary table.

Logging:
  - A timestamped log file is written to  logs/experiment_round6_<YYYYMMDD_HHMMSS>.log
  - All console output is mirrored to the log file in real-time

Usage:
    python main_round6.py
"""

import io
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logger setup ─────────────────────────────────────────────────────────────
_RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"experiment_round6_{_RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("greenai_round6")

# ─── Round 6 model definitions ────────────────────────────────────────────────
ROUND6_RUNS = [
    {
        "name":   "YOLO26 Round 6",
        "script": ROOT / "yolo26_round6"       / "run.py",
        "cwd":    ROOT / "yolo26_round6",
    },
    {
        "name":   "CNN Round 6",
        "script": ROOT / "cnn_round6"          / "run.py",
        "cwd":    ROOT / "cnn_round6",
    },
    {
        "name":   "ViT (Transformers) Round 6",
        "script": ROOT / "transformers_round6"  / "run.py",
        "cwd":    ROOT / "transformers_round6",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _banner(text: str, width: int = 70) -> None:
    line = "═" * width
    log.info(line)
    log.info(f"  {text}")
    log.info(line)


def _stream_output(stream: io.BufferedReader, label: str) -> None:
    """
    Read subprocess stdout/stderr line-by-line, printing to console and
    writing each line to the log file via the logger.
    """
    for raw in stream:
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:
            line = repr(raw)
        log.debug("[%s] %s", label, line)
        print(f"[{label}] {line}", flush=True)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_model(model_run: dict) -> bool:
    """
    Run a single Round 6 script as a subprocess.
    stdout/stderr are streamed live to the console AND mirrored to the log.
    Returns True on success, False on failure.
    """
    name   = model_run["name"]
    script = model_run["script"]
    cwd    = model_run["cwd"]

    _banner(f"Running Round 6 model: {name}")
    log.info("Script : %s", script)
    log.info("CWD    : %s", cwd)

    if not script.exists():
        log.error("Script not found: %s", script)
        return False

    start = time.time()
    log.info("Started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    t = Thread(target=_stream_output, args=(proc.stdout, name), daemon=True)
    t.start()
    proc.wait()
    t.join()

    elapsed = time.time() - start

    if proc.returncode == 0:
        log.info("[OK] %s finished in %.1f min (exit code 0)", name, elapsed / 60)
        return True
    else:
        log.error(
            "[FAIL] %s exited with code %d after %.1f min",
            name, proc.returncode, elapsed / 60,
        )
        return False


def run_report(script_path: Path, label: str) -> None:
    """Run a report script as a subprocess, mirroring output to the log."""
    if not script_path.exists():
        log.warning("%s not found at %s", label, script_path)
        return
    log.info("Running %s ...", label)
    rp = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    for line in (rp.stdout or "").splitlines():
        log.info("[%s] %s", label, line)
    for line in (rp.stderr or "").splitlines():
        log.warning("[%s] %s", label, line)
    if rp.returncode != 0:
        log.warning("%s exited with code %d", label, rp.returncode)
    else:
        log.info("%s completed successfully.", label)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("GreenAI Experiment — Sequential Round 6 Runner (30 Epoch Baseline — Final Round)", width=70)
    log.info("Log file  : %s", LOG_FILE)
    log.info("Run ID    : %s", _RUN_TS)
    log.info("Order     : YOLO26 Round 6 → CNN Round 6 → ViT Round 6")
    log.info("Goal      : Baseline parameters + 30 epochs (final experiment)")

    experiment_start = time.time()
    statuses: list[tuple[str, bool, float]] = []

    for model_run in ROUND6_RUNS:
        t0      = time.time()
        success = run_model(model_run)
        elapsed = (time.time() - t0) / 60
        statuses.append((model_run["name"], success, elapsed))

        if not success:
            log.warning("%s failed. Continuing with next run.", model_run["name"])

    # ── Generate Round 6 summary report ──────────────────────────────────────
    _banner("Generating Round 6 Summary Report", width=70)
    run_report(ROOT / "report_round6.py", "report_round6")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - experiment_start
    _banner("Round 6 Experiment Summary", width=70)

    log.info("%-30s %-10s %s", "Model", "Status", "Duration")
    log.info("-" * 55)
    for name, ok, elapsed_min in statuses:
        status_str = "OK" if ok else "FAILED"
        log.info("%-30s %-10s %.1f min", name, status_str, elapsed_min)
    log.info("-" * 55)
    log.info("Total experiment time : %.1f min", total_elapsed / 60)
    log.info("Log file              : %s", LOG_FILE)
    log.info("Round 6 report        : %s", ROOT / "reports" / "summary_report_round6.md")

    all_ok = all(ok for _, ok, _ in statuses)
    if not all_ok:
        log.warning("One or more Round 6 runs FAILED. Review the log above.")
        sys.exit(1)
    else:
        log.info("All Round 6 runs completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
