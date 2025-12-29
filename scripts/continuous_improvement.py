#!/usr/bin/env python3
"""Continuous improvement loop: test strategies, update dashboard, iterate.

Usage:
    poetry run python scripts/continuous_improvement.py [IMAGE_DIR] [--max-iterations N]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("continuous_improvement")


def main():
    parser = argparse.ArgumentParser(description="Continuous model improvement")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum improvement iterations")
    args = parser.parse_args()
    
    image_dir = args.image_dir or "~/Pictures/photo-dataset"
    
    log.info("="*60)
    log.info("CONTINUOUS MODEL IMPROVEMENT")
    log.info("="*60)
    
    scripts_dir = Path(__file__).resolve().parent
    
    # Run improvement strategies
    strategies = [
        ("retrain_improved_model.py", "Testing improved hyperparameters"),
        ("test_feature_combinations.py", "Testing feature combinations"),
    ]
    
    for script_name, description in strategies:
        log.info(f"\n{description}...")
        script_path = scripts_dir / script_name
        if script_path.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), image_dir],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    log.info(f"✓ {description} completed")
                    # Show key metrics
                    for line in result.stdout.split("\n"):
                        if any(keyword in line.lower() for keyword in ["accuracy", "improvement", "best"]):
                            log.info(f"  {line}")
                else:
                    log.warning(f"✗ {description} failed: {result.stderr[:200]}")
            except Exception as e:
                log.warning(f"✗ {description} error: {e}")
    
    # Update dashboard
    log.info("\nUpdating dashboard with latest results...")
    update_script = scripts_dir / "update_dashboard.py"
    if update_script.exists():
        try:
            subprocess.run([sys.executable, str(update_script)], check=True, timeout=60)
            log.info("✓ Dashboard updated")
        except Exception as e:
            log.warning(f"Dashboard update failed: {e}")
    
    log.info("\n" + "="*60)
    log.info("Improvement cycle complete!")
    log.info("Check .cache/summary_report/summary_report.html for updated dashboard")
    log.info("="*60)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

