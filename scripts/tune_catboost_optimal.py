#!/usr/bin/env python3
"""Tune CatBoost hyperparameters using Optuna for optimal performance.

Usage:
    poetry run python scripts/tune_catboost_optimal.py [IMAGE_DIR] [--n-trials N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("tune_catboost")


def objective(trial, X, y, scale_pos_weight):
    """Optuna objective function for CatBoost hyperparameter tuning."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        raise
    
    # Suggest hyperparameters
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    # Create pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**params)),
    ])
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    
    return scores.mean()


def main():
    parser = argparse.ArgumentParser(description="Tune CatBoost hyperparameters")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    try:
        import optuna
    except ImportError:
        log.error("Optuna not available. Install with: poetry add optuna")
        return 1
    
    log.info("="*80)
    log.info("CATBOOST HYPERPARAMETER TUNING")
    log.info("="*80)
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info(f"\nLoading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"\nStarting Optuna optimization ({args.n_trials} trials)...")
    log.info("This may take 10-30 minutes...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, scale_pos_weight),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    log.info("\n" + "="*80)
    log.info("OPTIMIZATION RESULTS")
    log.info("="*80)
    log.info(f"Best CV accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    log.info(f"Best parameters:")
    for key, value in study.best_params.items():
        log.info(f"  {key}: {value}")
    
    # Save results
    results_path = Path(__file__).resolve().parent.parent / ".cache" / "catboost_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_cv_accuracy": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": args.n_trials,
        }, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

