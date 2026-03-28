#!/usr/bin/env python3
"""Script 09: Run the complete CasCrop pipeline end-to-end.

Executes all scripts in order:
1. Download data
2. Process and match data
3. Build graphs
4. Train all models
5. Evaluate all models
6. Generate case study
7. Generate figures
8. Generate tables

This is the single-script reproduction entry point.
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPTS = [
    ("01_download_data.py", "Downloading all datasets"),
    ("02_process_data.py", "Processing and matching data"),
    ("03_build_graphs.py", "Building county graphs"),
    ("04_train_all.py", "Training all models"),
    ("05_evaluate_all.py", "Evaluating and testing"),
    ("06_case_study.py", "Generating case study"),
    ("07_generate_figures.py", "Generating figures"),
    ("08_generate_tables.py", "Generating tables"),
]


def run_script(script_name: str, config_path: str) -> bool:
    """Run a pipeline script and return success status."""
    script_path = Path("scripts") / script_name
    cmd = [sys.executable, str(script_path), "--config", config_path]

    logger.info(f"Running: {script_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FAILED: {script_name}")
        logger.error(result.stderr[-1000:] if result.stderr else "No error output")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run full CasCrop pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from script number (1-8)",
    )
    parser.add_argument(
        "--stop-after", type=int, default=8,
        help="Stop after script number (1-8)",
    )
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"CasCrop Full Pipeline — Started at {start_time}")
    logger.info(f"Config: {args.config}")

    results = {}

    for i, (script_name, description) in enumerate(SCRIPTS, 1):
        if i < args.start_from:
            continue
        if i > args.stop_after:
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Phase {i}/8: {description}")
        logger.info(f"{'='*60}")

        step_start = datetime.now()
        success = run_script(script_name, args.config)
        step_time = datetime.now() - step_start

        results[script_name] = {
            "success": success,
            "duration": str(step_time),
        }

        if not success:
            logger.error(f"Pipeline failed at step {i}: {script_name}")
            logger.error("Fix the issue and re-run with --start-from {i}")
            break

        logger.info(f"Completed in {step_time}")

    total_time = datetime.now() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete. Total time: {total_time}")
    logger.info(f"{'='*60}")

    # Print summary
    for script, result in results.items():
        status = "✓" if result["success"] else "✗"
        logger.info(f"  {status} {script}: {result['duration']}")


if __name__ == "__main__":
    main()
