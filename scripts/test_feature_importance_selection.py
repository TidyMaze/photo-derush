#!/usr/bin/env python3
"""Test feature selection based on XGBoost importance to improve accuracy.

Strategy: Use feature importance to identify and keep only the most informative features.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("feat_importance")


def main():
    parser = argparse.ArgumentParser(description="Test feature importance-based selection")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("FEATURE IMPORTANCE-BASED SELECTION")
    log.info("="*80)
    
    # Build dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    # Split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Train baseline to get feature importances
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
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
    
    log.info("Training baseline to get feature importances...")
    clf.fit(X_train, y_train)
    
    # Get feature importances
    xgb_model = clf.named_steps["xgb"]
    importances = xgb_model.feature_importances_
    
    # Sort features by importance
    feature_indices = np.argsort(importances)[::-1]
    
    log.info("\nTop 20 most important features:")
    for i, idx in enumerate(feature_indices[:20], 1):
        log.info(f"  {i}. Feature {idx}: {importances[idx]:.6f}")
    
    # Test different numbers of top features
    log.info("\n" + "="*80)
    log.info("TESTING DIFFERENT FEATURE COUNTS")
    log.info("="*80)
    
    baseline_acc = accuracy_score(y_test, clf.predict(X_test))
    log.info(f"\nBaseline (all {X.shape[1]} features): {baseline_acc:.4f}")
    
    results = []
    k_values = [X.shape[1], int(X.shape[1] * 0.9), int(X.shape[1] * 0.8), int(X.shape[1] * 0.7), int(X.shape[1] * 0.6), int(X.shape[1] * 0.5), 50, 40, 30, 20, 15, 10]
    k_values = sorted(set([k for k in k_values if k >= 10]), reverse=True)
    
    for k in k_values:
        top_k_indices = feature_indices[:k]
        X_train_k = X_train[:, top_k_indices]
        X_test_k = X_test[:, top_k_indices]
        
        clf_k = Pipeline([
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
        
        clf_k.fit(X_train_k, y_train)
        y_pred = clf_k.predict(X_test_k)
        acc = accuracy_score(y_test, y_pred)
        acc_diff = acc - baseline_acc
        
        log.info(f"Top {k:2d} features: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
        
        results.append({
            "k": k,
            "accuracy": float(acc),
            "accuracy_diff": float(acc_diff),
        })
        
        if acc_diff > 0.001:
            log.info(f"  ✅ IMPROVEMENT!")
    
    # Find best
    best_result = max(results, key=lambda x: x["accuracy"])
    
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    log.info(f"Baseline: {baseline_acc:.4f} (all {X.shape[1]} features)")
    log.info(f"Best: {best_result['accuracy']:.4f} (top {best_result['k']} features)")
    log.info(f"Improvement: {best_result['accuracy_diff']*100:+.2f}%")
    
    # Save results
    output_path = ".cache/feature_importance_selection_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "baseline_accuracy": float(baseline_acc),
            "baseline_n_features": X.shape[1],
            "best_k": best_result["k"],
            "best_accuracy": best_result["accuracy"],
            "improvement": best_result["accuracy_diff"],
            "results": results,
            "top_features": [int(idx) for idx in feature_indices[:best_result["k"]]],
        }, f, indent=2)
    
    log.info(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

