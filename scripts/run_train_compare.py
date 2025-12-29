#!/usr/bin/env python3
"""Run baseline training, tune hyperparameters (quick), retrain with tuned params,
and print a concise comparison report.

Usage:
  poetry run python scripts/run_train_compare.py [IMAGE_DIR]

Defaults to: /Users/yannrolland/Pictures/photo-dataset (as commonly used locally).
"""
from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("run_train_compare")

DEFAULT_DIR = os.path.expanduser("~/Pictures/photo-dataset")


def summarize(result):
    if result is None:
        return "(no result)"
    return (
        f"n_samples={result.n_samples}, n_keep={result.n_keep}, n_trash={result.n_trash},"
        f" cv_acc={result.cv_accuracy_mean:.4f}+/-{(result.cv_accuracy_std or 0):.4f},"
        f" precision={result.precision:.4f}, roc_auc={result.roc_auc:.4f}, f1={result.f1:.4f}"
    )


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Run baseline/train/tune comparison")
    p.add_argument("image_dir", nargs="?", default=DEFAULT_DIR)
    p.add_argument("--fast", action="store_true", help="Run a faster tuning (n_iter=6, cv_folds=2)")
    return p.parse_args()


def main():
    args = parse_args()
    image_dir = os.path.expanduser(args.image_dir)
    if not os.path.isdir(image_dir):
        log.error("Image directory does not exist: %s", image_dir)
        sys.exit(2)

    try:
        from src.training_core import train_keep_trash_model, DEFAULT_MODEL_PATH
        from src.tuning import tune_hyperparameters, load_best_params
    except Exception as e:
        log.exception("Import error: %s", e)
        sys.exit(2)

    log.info("Starting baseline training...")
    t0 = time.perf_counter()
    baseline = train_keep_trash_model(image_dir=image_dir, model_path="/tmp/photo_baseline.joblib", n_estimators=200, min_samples=2)
    t1 = time.perf_counter()
    log.info("Baseline training finished in %.1fs", t1 - t0)
    print("\nBASELINE:\n", summarize(baseline))

    if getattr(args, 'fast', False):
        n_iter = 6
        cv_folds = 2
        log.info("Running FAST hyperparameter tuning (n_iter=%d, cv_folds=%d)...", n_iter, cv_folds)
    else:
        n_iter = 10
        cv_folds = 3
        log.info("Running hyperparameter tuning (n_iter=%d, cv_folds=%d)...", n_iter, cv_folds)

    t0 = time.perf_counter()
    tuned = tune_hyperparameters(image_dir=image_dir, n_iter=n_iter, cv_folds=cv_folds, random_state=42, save_params=True)
    t1 = time.perf_counter()
    log.info("Tuning finished in %.1fs", t1 - t0)
    print("\nTUNED PARAMS:\n", tuned)

    # Show loaded best params to confirm
    best = load_best_params()
    print("\nLOADED BEST PARAMS FROM disk:\n", best)

    log.info("Retraining with tuned parameters (if any)...")
    t0 = time.perf_counter()
    tuned_result = train_keep_trash_model(image_dir=image_dir, model_path="/tmp/photo_tuned.joblib", n_estimators=200, min_samples=2)
    t1 = time.perf_counter()
    log.info("Tuned retrain finished in %.1fs", t1 - t0)
    print("\nTUNED RETRAIN:\n", summarize(tuned_result))

    # Simple compare
    print("\nCOMPARISON:\n")
    print("Baseline:", summarize(baseline))
    print("Tuned retrain:", summarize(tuned_result))

    # Exit with success
    sys.exit(0)


if __name__ == '__main__':
    main()
