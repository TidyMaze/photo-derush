#!/usr/bin/env python3
"""Optimize decision threshold for better accuracy.

Usage:
    poetry run python scripts/optimize_threshold.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("optimize_threshold")


def optimize_threshold(y_true, y_proba):
    """Find optimal threshold that maximizes accuracy."""
    thresholds = np.arange(0.3, 0.71, 0.01)
    best_threshold = 0.5
    best_accuracy = 0.0
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Optimize decision threshold")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("THRESHOLD OPTIMIZATION")
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
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Split train into train/validation for threshold optimization
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    log.info(f"Train: {len(y_train_split)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    
    # Train CatBoost model
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return 1
    
    n_keep = int(np.sum(y_train_split == 1))
    n_trash = int(np.sum(y_train_split == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )),
    ])
    
    log.info("\nTraining model...")
    clf.fit(X_train_split, y_train_split)
    
    # Get probabilities on validation set
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    
    # Optimize threshold
    log.info("\nOptimizing threshold on validation set...")
    best_threshold, best_accuracy, results = optimize_threshold(y_val, y_val_proba)
    
    log.info(f"\nOptimal threshold: {best_threshold:.3f}")
    log.info(f"Validation accuracy with optimal threshold: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Compare with default threshold
    y_val_pred_default = (y_val_proba >= 0.5).astype(int)
    default_accuracy = accuracy_score(y_val, y_val_pred_default)
    log.info(f"Default threshold (0.5) accuracy: {default_accuracy:.4f} ({default_accuracy*100:.2f}%)")
    log.info(f"Improvement: +{(best_accuracy - default_accuracy)*100:.2f} percentage points")
    
    # Evaluate on test set with optimal threshold
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    
    optimal_accuracy = accuracy_score(y_test, y_test_pred_optimal)
    default_test_accuracy = accuracy_score(y_test, y_test_pred_default)
    
    optimal_precision = precision_score(y_test, y_test_pred_optimal, zero_division=0)
    optimal_recall = recall_score(y_test, y_test_pred_optimal, zero_division=0)
    optimal_f1 = f1_score(y_test, y_test_pred_optimal, zero_division=0)
    optimal_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_optimal).ravel()
    
    log.info("\n" + "="*80)
    log.info("TEST SET RESULTS")
    log.info("="*80)
    log.info(f"Default threshold (0.5): {default_test_accuracy:.4f} ({default_test_accuracy*100:.2f}%)")
    log.info(f"Optimal threshold ({best_threshold:.3f}): {optimal_accuracy:.4f} ({optimal_accuracy*100:.2f}%)")
    log.info(f"Improvement: +{(optimal_accuracy - default_test_accuracy)*100:.2f} percentage points")
    log.info(f"\nMetrics with optimal threshold:")
    log.info(f"  Precision: {optimal_precision:.4f}")
    log.info(f"  Recall: {optimal_recall:.4f}")
    log.info(f"  F1: {optimal_f1:.4f}")
    log.info(f"  ROC-AUC: {optimal_roc_auc:.4f}")
    log.info(f"\nConfusion Matrix:")
    log.info(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    # Save results
    results_path = Path(__file__).resolve().parent.parent / ".cache" / "threshold_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "optimal_threshold": float(best_threshold),
            "validation_accuracy": float(best_accuracy),
            "default_accuracy": float(default_accuracy),
            "test_accuracy_default": float(default_test_accuracy),
            "test_accuracy_optimal": float(optimal_accuracy),
            "improvement": float(optimal_accuracy - default_test_accuracy),
            "test_metrics": {
                "accuracy": float(optimal_accuracy),
                "precision": float(optimal_precision),
                "recall": float(optimal_recall),
                "f1": float(optimal_f1),
                "roc_auc": float(optimal_roc_auc),
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
            "threshold_analysis": results,
        }, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

