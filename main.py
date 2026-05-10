"""
main.py — Orchestrator for the GreenAI experiment.

Runs the three baselines in sequence:
    1. YOLO26  (yolo26_baseline/run.py)   — also triggers Fashion-MNIST download
    2. CNN     (cnn_baseline/run.py)
    3. ViT     (transformers_baseline/run.py)

Each baseline:
  - Manages its own CodeCarbon tracker internally
  - Writes artifacts under  <baseline>/output/runs/train/
  - Saves an emissions.csv   under  <baseline>/output/

After all baselines, report.py is called to generate a summary table.

Logging:
  - A timestamped log file is written to  logs/experiment_<YYYYMMDD_HHMMSS>.log
  - All console output is mirrored to the log file in real-time

Usage:
    python main.py

Exit codes:
    0  — all baselines completed successfully
    1  — one or more baselines failed (check log for details)
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
LOG_FILE = LOGS_DIR / f"experiment_{_RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("greenai")

# ─── Baseline definitions ─────────────────────────────────────────────────────
BASELINES = [
    {
        "name":   "YOLO26",
        "script": ROOT / "yolo26_baseline"      / "run.py",
        "cwd":    ROOT / "yolo26_baseline",
    },
    {
        "name":   "CNN",
        "script": ROOT / "cnn_baseline"         / "run.py",
        "cwd":    ROOT / "cnn_baseline",
    },
    {
        "name":   "ViT (Transformers)",
        "script": ROOT / "transformers_baseline" / "run.py",
        "cwd":    ROOT / "transformers_baseline",
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
    Runs in a daemon thread so it doesn't block the main process.
    """
    for raw in stream:
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:
            line = repr(raw)
        # log at DEBUG so the subprocess output appears in the file but can
        # be filtered separately from orchestrator messages.
        log.debug("[%s] %s", label, line)
        # Also echo directly to stdout so the user sees live training progress.
        print(f"[{label}] {line}", flush=True)


# ─── Baseline runner ──────────────────────────────────────────────────────────

def run_baseline(baseline: dict) -> bool:
    """
    Run a single baseline script as a subprocess.
    stdout/stderr are streamed live to the console AND mirrored to the log.
    Returns True on success, False on failure.
    """
    name   = baseline["name"]
    script = baseline["script"]
    cwd    = baseline["cwd"]

    _banner(f"Running baseline: {name}")
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
        stderr=subprocess.STDOUT,   # merge stderr into stdout
    )

    # Stream output in a background thread so the main thread can wait()
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("GreenAI Experiment — Sequential Baseline Runner", width=70)
    log.info("Log file  : %s", LOG_FILE)
    log.info("Run ID    : %s", _RUN_TS)
    log.info("Order     : YOLO26 → CNN → ViT (Transformers)")
    log.info("Note      : YOLO26 runs first to download the Fashion-MNIST dataset.")

    experiment_start = time.time()
    statuses: list[tuple[str, bool, float]] = []   # (name, ok, elapsed_min)

    for baseline in BASELINES:
        t0      = time.time()
        success = run_baseline(baseline)
        elapsed = (time.time() - t0) / 60
        statuses.append((baseline["name"], success, elapsed))

        if not success:
            log.warning("%s failed. Continuing with next baseline.", baseline["name"])

    # ── Generate summary report ───────────────────────────────────────────────
    _banner("Generating Summary Report", width=70)
    report_script = ROOT / "report.py"
    if report_script.exists():
        log.info("Running report.py ...")
        rp = subprocess.run(
            [sys.executable, str(report_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if rp.stdout:
            for line in rp.stdout.splitlines():
                log.info("[report] %s", line)
        if rp.stderr:
            for line in rp.stderr.splitlines():
                log.warning("[report] %s", line)
        if rp.returncode != 0:
            log.warning("report.py exited with code %d", rp.returncode)
        else:
            log.info("report.py completed successfully.")
    else:
        log.warning("report.py not found at %s", report_script)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - experiment_start
    _banner("Experiment Summary", width=70)

    log.info("%-25s %-10s %s", "Baseline", "Status", "Duration")
    log.info("-" * 50)
    for name, ok, elapsed_min in statuses:
        status_str = "OK" if ok else "FAILED"
        log.info("%-25s %-10s %.1f min", name, status_str, elapsed_min)
    log.info("-" * 50)
    log.info("Total experiment time : %.1f min", total_elapsed / 60)
    log.info("Log file              : %s", LOG_FILE)
    log.info("Summary report        : %s", ROOT / "reports" / "summary_report.md")

    all_ok = all(ok for _, ok, _ in statuses)
    if not all_ok:
        log.warning("One or more baselines FAILED. Review the log above.")
        sys.exit(1)
    else:
        log.info("All baselines completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
