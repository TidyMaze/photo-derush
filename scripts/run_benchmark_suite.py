#!/usr/bin/env python3
"""Run complete benchmark suite and generate comparison reports.

Usage:
    poetry run python scripts/run_benchmark_suite.py [IMAGE_DIR] [--skip SKIP_LIST]
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("benchmark_suite")

SCRIPTS_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def run_script(script_name: str, image_dir: str, args: list[str] | None = None) -> bool:
    """Run a benchmark script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        log.warning(f"Script not found: {script_name}")
        return False

    cmd = [sys.executable, str(script_path), image_dir] + (args or [])
    log.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        if result.returncode == 0:
            log.info(f"✓ {script_name} completed successfully")
            return True
        else:
            log.error(f"✗ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                log.error(f"Error output: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"✗ {script_name} timed out")
        return False
    except Exception as e:
        log.error(f"✗ {script_name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--skip", nargs="+", default=[], help="Scripts to skip (e.g., --skip augmentation calibration)")
    parser.add_argument("--only", nargs="+", default=None, help="Only run these scripts")
    args = parser.parse_args()

    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")

    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1

    log.info(f"Running benchmark suite on: {image_dir}")
    log.info(f"Cache directory: {CACHE_DIR}")

    # Benchmark scripts in order
    benchmarks = [
        ("benchmark_model.py", ["--baseline"], "Baseline benchmark"),
        ("feature_ablation.py", [], "Feature ablation study"),
        ("compare_algorithms.py", [], "Algorithm comparison"),
        ("ensemble_experiments.py", [], "Ensemble experiments"),
        ("calibration_study.py", [], "Calibration study"),
        ("class_balance_study.py", [], "Class balance study"),
        ("augmentation_study.py", [], "Augmentation study"),
        ("full_evaluation.py", [], "Full evaluation"),
    ]

    # Filter by --only or --skip
    if args.only:
        benchmarks = [(name, script_args, desc) for name, script_args, desc in benchmarks if name.replace(".py", "") in args.only]
    else:
        benchmarks = [(name, script_args, desc) for name, script_args, desc in benchmarks if name.replace(".py", "") not in args.skip]

    results = {}
    for script_name, script_args, description in benchmarks:
        log.info(f"\n{'='*60}")
        log.info(f"Running: {description}")
        log.info(f"{'='*60}")
        success = run_script(script_name, image_dir, script_args)
        results[script_name] = success

    # Generate dashboard
    log.info(f"\n{'='*60}")
    log.info("Generating dashboard...")
    log.info(f"{'='*60}")

    dashboard_script = SCRIPTS_DIR / "generate_benchmark_dashboard.py"
    if dashboard_script.exists():
        cmd = [sys.executable, str(dashboard_script)]
        try:
            subprocess.run(cmd, check=True, timeout=60)
            log.info("✓ Dashboard generated successfully")
        except Exception as e:
            log.warning(f"Dashboard generation failed: {e}")
    else:
        log.warning("Dashboard script not found")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("Benchmark Suite Summary")
    log.info(f"{'='*60}")

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    for script_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        log.info(f"{status}: {script_name}")

    log.info(f"\nCompleted: {successful}/{total} benchmarks")
    log.info(f"Results saved in: {CACHE_DIR}")
    log.info(f"Dashboard: {CACHE_DIR / 'benchmark_dashboard.html'}")

    return 0 if successful == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

