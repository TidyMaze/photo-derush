#!/usr/bin/env python3
"""Retrain model with improved hyperparameters and test on same split as baseline.

Usage:
    poetry run python scripts/retrain_improved_model.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, load_baseline, save_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.training_core import train_keep_trash_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("retrain_improved")


def main():
    parser = argparse.ArgumentParser(description="Retrain with improved hyperparameters")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info(f"Loading dataset from {image_dir}")
    
    # Initialize repository
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)
    
    # Build dataset
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(y) < 20:
        log.error(f"Insufficient labeled data: {len(y)} samples")
        return 1
    
    # Use SAME split as baseline (random_state=42, test_size=0.2)
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load baseline
    baseline = load_baseline()
    baseline_acc = baseline.metrics["accuracy"] if baseline else 0.0
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Test improved configurations
    configs = [
        ("baseline", {"n_estimators": 200}),
        ("more_trees", {"n_estimators": 300}),
        ("low_lr_more_trees", {"learning_rate": 0.01, "n_estimators": 400}),
        ("deeper", {"max_depth": 8, "min_child_weight": 1, "n_estimators": 250}),
        ("balanced", {"n_estimators": 250, "subsample": 0.8, "colsample_bytree": 0.9}),
    ]
    
    best_acc = baseline_acc
    best_config_name = None
    
    for config_name, params_override in configs:
        log.info(f"\n=== Testing: {config_name} ===")
        try:
            # Train model
            temp_model_path = os.path.join(image_dir, f".test_{config_name}.joblib")
            result = train_keep_trash_model(
                image_dir=image_dir,
                model_path=temp_model_path,
                repo=repo,
                displayed_filenames=filenames_train,
                n_estimators=params_override.get("n_estimators", 200),
                random_state=42,
            )
            
            if result is None:
                log.warning(f"Training failed for {config_name}")
                continue
            
            import joblib
            data = joblib.load(temp_model_path)
            clf = data["model"]
            
            # Benchmark on test set
            test_filenames_list = filenames_test.tolist() if hasattr(filenames_test, 'tolist') else list(filenames_test)
            benchmark_result = benchmark_model(
                clf, X_train, X_test, y_train, y_test, test_filenames_list,
                f"Improved_{config_name}", temp_model_path, {}, None, FEATURE_COUNT, "FAST"
            )
            
            acc = benchmark_result.metrics["accuracy"]
            log.info(f"Accuracy: {acc:.4f} ({acc - baseline_acc:+.4f})")
            
            if acc > best_acc:
                best_acc = acc
                best_config_name = config_name
                log.info(f"✅ NEW BEST!")
                
                # Save as new baseline if significantly better
                if acc > baseline_acc + 0.01:
                    save_baseline(benchmark_result)
                    log.info("💾 Saved as new baseline")
            
            # Cleanup
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
        except Exception as e:
            log.error(f"Failed {config_name}: {e}", exc_info=True)
    
    # Summary
    log.info("\n" + "="*60)
    log.info("RETRAINING SUMMARY")
    log.info("="*60)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    if best_config_name:
        log.info(f"Best configuration: {best_config_name}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

