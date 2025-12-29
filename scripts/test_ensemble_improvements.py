#!/usr/bin/env python3
"""Test ensemble methods with best feature set and tuned hyperparameters.

Usage:
    poetry run python scripts/test_ensemble_improvements.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, load_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("ensemble_improvements")


def get_best_feature_indices():
    """Get best feature indices from feature combination results."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    combo_path = cache_dir / "feature_combination_results.json"
    
    if combo_path.exists():
        try:
            with open(combo_path) as f:
                combo_data = json.load(f)
                best_config = combo_data.get("best_config")
                if best_config:
                    return best_config.get("feature_indices", list(range(FEATURE_COUNT)))
        except Exception:
            pass
    
    # Default: top 20 features (from previous study)
    return list(range(min(20, FEATURE_COUNT)))


def test_ensemble(X_train, X_test, y_train, y_test, filenames_test, config_name, feature_indices):
    """Test an ensemble configuration."""
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Create base estimators
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        **xgb_params,
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("rf", rf_clf),
        ],
        voting="soft",
        weights=[2, 1],  # Weight XGBoost more
    )
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", ensemble),
    ])
    
    clf.fit(X_train_subset, y_train)
    
    test_filenames_list = filenames_test.tolist() if hasattr(filenames_test, 'tolist') else list(filenames_test)
    benchmark_result = benchmark_model(
        clf, X_train_subset, X_test_subset, y_train, y_test, test_filenames_list,
        config_name, None, xgb_params, None, len(feature_indices), "FAST"
    )
    
    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="Test ensemble improvements")
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
    
    # Load baseline
    baseline = load_baseline()
    baseline_acc = baseline.metrics["accuracy"] if baseline else 0.0
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Get best feature indices
    feature_indices = get_best_feature_indices()
    log.info(f"Using {len(feature_indices)} features")
    
    results = []
    best_acc = baseline_acc
    best_result = None
    
    # Test ensemble
    log.info("\n=== Testing Ensemble (XGBoost + Random Forest) ===")
    try:
        result = test_ensemble(X_train, X_test, y_train, y_test, filenames_test, "Ensemble_XGB_RF", feature_indices)
        results.append({"strategy": "ensemble_xgb_rf", "result": result})
        if result.metrics["accuracy"] > best_acc:
            best_acc = result.metrics["accuracy"]
            best_result = result
        log.info(f"Ensemble accuracy: {result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.warning(f"Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    log.info("\n" + "="*60)
    log.info("ENSEMBLE IMPROVEMENT SUMMARY")
    log.info("="*60)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    
    # Save results
    output_path = Path(__file__).resolve().parent.parent / ".cache" / "ensemble_improvements.json"
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

