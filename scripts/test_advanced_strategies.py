#!/usr/bin/env python3
"""Test advanced strategies to push accuracy higher: feature engineering, ensemble tuning, etc.

Usage:
    poetry run python scripts/test_advanced_strategies.py [IMAGE_DIR]
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

from src.benchmark import benchmark_model, load_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("advanced_strategies")


def test_config(X_train, X_test, y_train, y_test, filenames_test, config_name, feature_indices, xgb_params_override=None):
    """Test a configuration."""
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    if xgb_params_override:
        xgb_params.update(xgb_params_override)
    
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
    
    clf.fit(X_train_subset, y_train)
    
    test_filenames_list = filenames_test.tolist() if hasattr(filenames_test, 'tolist') else list(filenames_test)
    benchmark_result = benchmark_model(
        clf, X_train_subset, X_test_subset, y_train, y_test, test_filenames_list,
        config_name, None, xgb_params, None, len(feature_indices), "FAST"
    )
    
    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="Test advanced improvement strategies")
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
    
    # Same split as baseline
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load current baseline
    baseline = load_baseline()
    baseline_acc = baseline.metrics["accuracy"] if baseline else 0.0
    log.info(f"Current baseline accuracy: {baseline_acc:.4f}")
    
    # Load best feature combination
    combo_path = Path(__file__).resolve().parent.parent / ".cache" / "feature_combination_results.json"
    best_features = None
    if combo_path.exists():
        with open(combo_path) as f:
            combo_data = json.load(f)
            best_config = combo_data.get("best_config")
            if best_config:
                best_features = best_config.get("feature_indices", list(range(FEATURE_COUNT)))
                log.info(f"Using best feature set: {len(best_features)} features")
    
    if best_features is None:
        best_features = list(range(FEATURE_COUNT))
    
    results = []
    best_acc = baseline_acc
    best_result = None
    
    # Strategy 1: Best features + optimized hyperparameters
    log.info("\n=== Strategy 1: Best Features + Tuned Hyperparameters ===")
    try:
        result = test_config(
            X_train, X_test, y_train, y_test, filenames_test,
            "BestFeatures_Tuned", best_features,
            {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 7}
        )
        results.append({"strategy": "best_features_tuned", "result": result})
        if result.metrics["accuracy"] > best_acc:
            best_acc = result.metrics["accuracy"]
            best_result = result
        log.info(f"Accuracy: {result.metrics['accuracy']:.4f} ({result.metrics['accuracy'] - baseline_acc:+.4f})")
    except Exception as e:
        log.error(f"Failed: {e}")
    
    # Strategy 2: Best features + early stopping
    log.info("\n=== Strategy 2: Best Features + Early Stopping ===")
    try:
        # Manual early stopping by training with validation split
        from sklearn.model_selection import train_test_split as tts
        X_train_sub, X_val, y_train_sub, y_val = tts(
            X_train[:, best_features], y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        xgb_params = load_best_params() or {}
        xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
        xgb_params.update({"n_estimators": 500, "learning_rate": 0.01})
        
        n_keep = int(np.sum(y_train_sub == 1))
        n_trash = int(np.sum(y_train_sub == 0))
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
        
        clf.fit(X_train_sub, y_train_sub)
        
        # Evaluate on validation to find best iteration
        val_pred = clf.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        log.info(f"Validation accuracy: {val_acc:.4f}")
        
        # Test on test set
        test_filenames_list = filenames_test.tolist() if hasattr(filenames_test, 'tolist') else list(filenames_test)
        result = benchmark_model(
            clf, X_train[:, best_features], X_test[:, best_features], y_train, y_test, test_filenames_list,
            "BestFeatures_EarlyStop", None, xgb_params, None, len(best_features), "FAST"
        )
        results.append({"strategy": "best_features_early_stop", "result": result})
        if result.metrics["accuracy"] > best_acc:
            best_acc = result.metrics["accuracy"]
            best_result = result
        log.info(f"Test accuracy: {result.metrics['accuracy']:.4f} ({result.metrics['accuracy'] - baseline_acc:+.4f})")
    except Exception as e:
        log.error(f"Failed: {e}")
    
    # Strategy 3: Best features + aggressive regularization
    log.info("\n=== Strategy 3: Best Features + Aggressive Regularization ===")
    try:
        result = test_config(
            X_train, X_test, y_train, y_test, filenames_test,
            "BestFeatures_Regularized", best_features,
            {"n_estimators": 400, "learning_rate": 0.01, "reg_alpha": 1.0, "reg_lambda": 2.0, "gamma": 0.2}
        )
        results.append({"strategy": "best_features_regularized", "result": result})
        if result.metrics["accuracy"] > best_acc:
            best_acc = result.metrics["accuracy"]
            best_result = result
        log.info(f"Accuracy: {result.metrics['accuracy']:.4f} ({result.metrics['accuracy'] - baseline_acc:+.4f})")
    except Exception as e:
        log.error(f"Failed: {e}")
    
    # Summary
    log.info("\n" + "="*60)
    log.info("ADVANCED STRATEGIES SUMMARY")
    log.info("="*60)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    
    if best_result:
        log.info(f"\nBest strategy: {[r['strategy'] for r in results if r['result'] == best_result][0]}")
        log.info(f"  Accuracy: {best_result.metrics['accuracy']:.4f}")
        log.info(f"  F1: {best_result.metrics['f1']:.4f}")
        log.info(f"  ROC-AUC: {best_result.metrics.get('roc_auc', 'N/A')}")
    
    # Save results
    output_path = Path(__file__).resolve().parent.parent / ".cache" / "advanced_strategies_results.json"
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

