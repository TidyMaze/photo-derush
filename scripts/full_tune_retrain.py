#!/usr/bin/env python3
"""Run a more thorough hyperparameter tuning (n_iter=30, cv_folds=5), save tuned params, retrain model and calibrator, and report metrics.

Usage:
  poetry run python scripts/full_tune_retrain.py [IMAGE_DIR]
"""
from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("full_tune_retrain")

DEFAULT_DIR = os.path.expanduser("~/Pictures/photo-dataset")


def main():
    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        log.error("Image directory does not exist: %s", image_dir)
        sys.exit(2)

    try:
        from src.tuning import tune_hyperparameters, load_best_params
        from src.training_core import train_keep_trash_model
    except Exception as e:
        log.exception("Import error: %s", e)
        sys.exit(2)

    log.info("Starting extensive tuning (n_iter=30, cv_folds=5)...")
    t0 = time.perf_counter()
    best = tune_hyperparameters(image_dir=image_dir, n_iter=30, cv_folds=5, random_state=42, save_params=True)
    t1 = time.perf_counter()
    log.info("Tuning finished in %.1fs", t1 - t0)
    print("TUNED PARAMS:\n", best)

    loaded = load_best_params()
    print("LOADED BEST PARAMS:\n", loaded)

    log.info("Retraining final model with tuned params...")
    t0 = time.perf_counter()
    res = train_keep_trash_model(image_dir=image_dir, model_path="/tmp/photo_final.joblib", n_estimators=200, min_samples=2)
    t1 = time.perf_counter()
    log.info("Retrain finished in %.1fs", t1 - t0)
    print("TRAINING RESULT:\n", res)
    sys.exit(0)


if __name__ == '__main__':
    main()
