#!/usr/bin/env python3
"""Comprehensive model improvement: test multiple strategies and update dashboard.

Usage:
    poetry run python scripts/comprehensive_improvement.py [IMAGE_DIR]
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, load_baseline, save_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params, tune_hyperparameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("comprehensive_improvement")


def test_model_config(X_train, X_test, y_train, y_test, filenames_test, config_name, params_override=None):
    """Test a model configuration."""
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    if params_override:
        xgb_params.update(params_override)
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            **xgb_params,
        )),
    ])
    
    clf.fit(X_train, y_train)
    
    benchmark_result = benchmark_model(
        clf, X_train, X_test, y_train, y_test, filenames_test,
        config_name, None, xgb_params, None, FEATURE_COUNT, "FAST"
    )
    
    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model improvement")
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
    
    # Fixed split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load baseline or compute it
    baseline = load_baseline()
    if baseline:
        baseline_acc = baseline.metrics["accuracy"]
        log.info(f"Baseline accuracy (from saved): {baseline_acc:.4f}")
    else:
        # Compute baseline with current best params
        log.info("No baseline found, computing baseline...")
        baseline_result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "baseline")
        baseline_acc = baseline_result.metrics["accuracy"]
        log.info(f"Baseline accuracy (computed): {baseline_acc:.4f}")
    
    results = []
    best_acc = baseline_acc
    best_result = None
    
    # Strategy 1: Current best params
    log.info("\n=== Strategy 1: Current Best Hyperparameters ===")
    result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "CurrentBest")
    results.append({"strategy": "current_best", "result": result})
    if result.metrics["accuracy"] > best_acc:
        best_acc = result.metrics["accuracy"]
        best_result = result
    log.info(f"Accuracy: {result.metrics['accuracy']:.4f}")
    
    # Strategy 2: Hyperparameter tuning
    log.info("\n=== Strategy 2: Hyperparameter Tuning ===")
    try:
        tuned_params = tune_hyperparameters(
            image_dir=image_dir,
            repo=repo,
            n_iter=30,
            cv_folds=3,
            random_state=42,
            save_params=False,
        )
        
        if tuned_params:
            result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "TunedParams", tuned_params)
            results.append({"strategy": "tuned_params", "result": result, "params": tuned_params})
            if result.metrics["accuracy"] > best_acc:
                best_acc = result.metrics["accuracy"]
                best_result = result
            log.info(f"Tuned accuracy: {result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.warning(f"Tuning failed: {e}")
    
    # Strategy 3: Increased n_estimators
    log.info("\n=== Strategy 3: More Trees ===")
    result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "MoreTrees", {"n_estimators": 300})
    results.append({"strategy": "more_trees", "result": result})
    if result.metrics["accuracy"] > best_acc:
        best_acc = result.metrics["accuracy"]
        best_result = result
    log.info(f"Accuracy: {result.metrics['accuracy']:.4f}")
    
    # Strategy 4: Lower learning rate + more trees
    log.info("\n=== Strategy 4: Lower LR + More Trees ===")
    result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "LowLRMoreTrees", {
        "learning_rate": 0.01,
        "n_estimators": 400,
    })
    results.append({"strategy": "low_lr_more_trees", "result": result})
    if result.metrics["accuracy"] > best_acc:
        best_acc = result.metrics["accuracy"]
        best_result = result
    log.info(f"Accuracy: {result.metrics['accuracy']:.4f}")
    
    # Strategy 5: Deeper trees
    log.info("\n=== Strategy 5: Deeper Trees ===")
    result = test_model_config(X_train, X_test, y_train, y_test, filenames_test, "DeeperTrees", {
        "max_depth": 8,
        "min_child_weight": 1,
    })
    results.append({"strategy": "deeper_trees", "result": result})
    if result.metrics["accuracy"] > best_acc:
        best_acc = result.metrics["accuracy"]
        best_result = result
    log.info(f"Accuracy: {result.metrics['accuracy']:.4f}")
    
    # Summary
    log.info("\n" + "="*60)
    log.info("IMPROVEMENT SUMMARY")
    log.info("="*60)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    if baseline_acc > 0:
        log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    else:
        log.info(f"Improvement: {best_acc - baseline_acc:+.4f} (baseline was 0, cannot compute percentage)")
    
    if best_result:
        log.info(f"\nBest strategy: {[r['strategy'] for r in results if r['result'] == best_result][0]}")
        log.info(f"  Accuracy: {best_result.metrics['accuracy']:.4f}")
        log.info(f"  F1: {best_result.metrics['f1']:.4f}")
        log.info(f"  ROC-AUC: {best_result.metrics.get('roc_auc', 'N/A')}")
        
        # Save as new baseline if improved
        if best_acc > baseline_acc + 0.01:  # At least 1% improvement
            log.info("\n💾 Saving improved model as new baseline...")
            save_baseline(best_result)
    
    # Save results
    output_path = Path(__file__).resolve().parent.parent / ".cache" / "comprehensive_improvement.json"
    with open(output_path, "w") as f:
        json.dump({
            "baseline_accuracy": baseline_acc,
            "best_accuracy": best_acc,
            "improvement": best_acc - baseline_acc,
            "results": [{"strategy": r["strategy"], "metrics": r["result"].metrics} for r in results],
        }, f, indent=2, default=str)
    
    log.info(f"\nSaved results to {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

