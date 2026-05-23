"""
main_modified.py — Orchestrator for the GreenAI modified experiment phase.

Runs the three modified models in sequence:
    1. YOLO26  (yolo26_modified/run.py)
    2. CNN     (cnn_modified/run.py)
    3. ViT     (transformers_modified/run.py)

Each modified run:
  - Manages its own CodeCarbon tracker internally
  - Writes artifacts under  <model>_modified/output/runs/train/
  - Saves an emissions.csv   under  <model>_modified/output/

After all runs, report_modified.py is called to generate a summary table.

Logging:
  - A timestamped log file is written to  logs/experiment_modified_<YYYYMMDD_HHMMSS>.log
  - All console output is mirrored to the log file in real-time

Usage:
    python main_modified.py

Exit codes:
    0  — all runs completed successfully
    1  — one or more runs failed (check log for details)
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
LOG_FILE = LOGS_DIR / f"experiment_modified_{_RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("greenai_modified")

# ─── Modified Model definitions ───────────────────────────────────────────────
MODIFIED_RUNS = [
    {
        "name":   "YOLO26 Modified",
        "script": ROOT / "yolo26_modified"      / "run.py",
        "cwd":    ROOT / "yolo26_modified",
    },
    {
        "name":   "CNN Modified",
        "script": ROOT / "cnn_modified"         / "run.py",
        "cwd":    ROOT / "cnn_modified",
    },
    {
        "name":   "ViT (Transformers) Modified",
        "script": ROOT / "transformers_modified" / "run.py",
        "cwd":    ROOT / "transformers_modified",
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
        log.debug("[%s] %s", label, line)
        print(f"[{label}] {line}", flush=True)


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_model(model_run: dict) -> bool:
    """
    Run a single modified script as a subprocess.
    stdout/stderr are streamed live to the console AND mirrored to the log.
    Returns True on success, False on failure.
    """
    name   = model_run["name"]
    script = model_run["script"]
    cwd    = model_run["cwd"]

    _banner(f"Running modified model: {name}")
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("GreenAI Experiment — Sequential Modified Runner", width=70)
    log.info("Log file  : %s", LOG_FILE)
    log.info("Run ID    : %s", _RUN_TS)
    log.info("Order     : YOLO26 Modified → CNN Modified → ViT Modified")

    experiment_start = time.time()
    statuses: list[tuple[str, bool, float]] = []

    for model_run in MODIFIED_RUNS:
        t0      = time.time()
        success = run_model(model_run)
        elapsed = (time.time() - t0) / 60
        statuses.append((model_run["name"], success, elapsed))

        if not success:
            log.warning("%s failed. Continuing with next run.", model_run["name"])

    # ── Generate summary report ───────────────────────────────────────────────
    _banner("Generating Summary Report", width=70)
    report_script = ROOT / "report_modified.py"
    if report_script.exists():
        log.info("Running report_modified.py ...")
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
            log.warning("report_modified.py exited with code %d", rp.returncode)
        else:
            log.info("report_modified.py completed successfully.")
    else:
        log.warning("report_modified.py not found at %s", report_script)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - experiment_start
    _banner("Modified Runs Experiment Summary", width=70)

    log.info("%-30s %-10s %s", "Model", "Status", "Duration")
    log.info("-" * 55)
    for name, ok, elapsed_min in statuses:
        status_str = "OK" if ok else "FAILED"
        log.info("%-30s %-10s %.1f min", name, status_str, elapsed_min)
    log.info("-" * 55)
    log.info("Total experiment time : %.1f min", total_elapsed / 60)
    log.info("Log file              : %s", LOG_FILE)
    log.info("Summary report        : %s", ROOT / "reports" / "summary_report_modified.md")

    all_ok = all(ok for _, ok, _ in statuses)
    if not all_ok:
        log.warning("One or more modified runs FAILED. Review the log above.")
        sys.exit(1)
    else:
        log.info("All modified runs completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
