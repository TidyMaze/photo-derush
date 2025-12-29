#!/usr/bin/env python3
"""Iteratively improve model by testing feature combinations and hyperparameters.

Usage:
    poetry run python scripts/improve_model.py [IMAGE_DIR] [--iterations N]
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

from src.benchmark import benchmark_model, save_baseline, load_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.tuning import load_best_params, tune_hyperparameters_optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("improve_model")


def test_feature_subset(X: np.ndarray, y: np.ndarray, feature_indices: list[int], test_name: str) -> dict:
    """Test model with subset of features."""
    X_subset = X[:, feature_indices]
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.2, stratify=y, random_state=42
    )
    
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
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    from src.benchmark import compute_metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    return {
        "test_name": test_name,
        "n_features": len(feature_indices),
        "feature_indices": feature_indices,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Iteratively improve model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--iterations", type=int, default=3, help="Number of improvement iterations")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna for hyperparameter tuning")
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
    
    log.info(f"Dataset: {len(y)} samples, {FEATURE_COUNT} features")
    
    # Load baseline
    baseline = load_baseline()
    baseline_acc = baseline.metrics["accuracy"] if baseline else 0.0
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    results = []
    best_acc = baseline_acc
    best_config = None
    
    # Iteration 1: Test with top features from feature selection
    log.info("\n=== Iteration 1: Testing Top Features ===")
    try:
        feature_selection_path = Path(__file__).resolve().parent.parent / ".cache" / "feature_selection_results.json"
        if feature_selection_path.exists():
            with open(feature_selection_path) as f:
                fs_data = json.load(f)
            
            # Get top features from XGBoost importance
            if "xgboost_importance" in fs_data:
                top_features = [idx for idx, _ in fs_data["xgboost_importance"][:30]]
                result = test_feature_subset(X, y, top_features, "top_30_features")
                results.append(result)
                if result["metrics"]["accuracy"] > best_acc:
                    best_acc = result["metrics"]["accuracy"]
                    best_config = result
                log.info(f"Top 30 features: Accuracy = {result['metrics']['accuracy']:.4f}")
    except Exception as e:
        log.warning(f"Feature selection test failed: {e}")
    
    # Iteration 2: Hyperparameter tuning with Optuna
    if args.use_optuna:
        log.info("\n=== Iteration 2: Optuna Hyperparameter Tuning ===")
        try:
            X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
                X, y, filenames, test_size=0.2, stratify=y, random_state=42
            )
            
            best_params = tune_hyperparameters_optuna(
                image_dir=image_dir,
                repo=repo,
                n_trials=30,
                cv_folds=3,
                random_state=42,
                save_params=False,
            )
            
            if best_params:
                # Test with optimized params
                scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("xgb", xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **{k: v for k, v in best_params.items() if not k.startswith("_")},
                    )),
                ])
                
                clf.fit(X_train, y_train)
                benchmark_result = benchmark_model(
                    clf, X_train, X_test, y_train, y_test, filenames_test,
                    "OptunaTuned", None, best_params, None, FEATURE_COUNT,
                    "FULL" if USE_FULL_FEATURES else "FAST"
                )
                
                results.append({
                    "test_name": "optuna_tuned",
                    "n_features": FEATURE_COUNT,
                    "metrics": benchmark_result.metrics,
                    "hyperparameters": best_params,
                })
                
                if benchmark_result.metrics["accuracy"] > best_acc:
                    best_acc = benchmark_result.metrics["accuracy"]
                    best_config = results[-1]
                
                log.info(f"Optuna tuned: Accuracy = {benchmark_result.metrics['accuracy']:.4f}")
        except Exception as e:
            log.warning(f"Optuna tuning failed: {e}")
    
    # Iteration 3: Test EXIF + Quality features (most important from ablation)
    log.info("\n=== Iteration 3: EXIF + Quality Features ===")
    # Based on ablation study, EXIF and Quality are most important
    # EXIF features are roughly indices 46-71 (for FAST mode)
    # Quality features are roughly indices 36-42
    exif_indices = list(range(46, min(71, FEATURE_COUNT)))
    quality_indices = list(range(36, min(42, FEATURE_COUNT)))
    combined_indices = sorted(set(exif_indices + quality_indices + list(range(12))))  # Add histograms
    
    result = test_feature_subset(X, y, combined_indices, "exif_quality_hist")
    results.append(result)
    if result["metrics"]["accuracy"] > best_acc:
        best_acc = result["metrics"]["accuracy"]
        best_config = result
    log.info(f"EXIF+Quality+Hist: Accuracy = {result['metrics']['accuracy']:.4f}")
    
    # Summary
    log.info("\n=== Improvement Summary ===")
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    
    if best_config:
        log.info(f"Best configuration: {best_config['test_name']}")
        log.info(f"  Features: {best_config.get('n_features', 'all')}")
        log.info(f"  Accuracy: {best_config['metrics']['accuracy']:.4f}")
        log.info(f"  F1: {best_config['metrics']['f1']:.4f}")
    
    # Save results
    output_path = Path(__file__).resolve().parent.parent / ".cache" / "model_improvement_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "baseline_accuracy": baseline_acc,
            "best_accuracy": best_acc,
            "improvement": best_acc - baseline_acc,
            "results": results,
            "best_config": best_config,
        }, f, indent=2, default=str)
    
    log.info(f"\nSaved results to {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

