#!/usr/bin/env python3
"""Remove features that contribute negatively to accuracy.

Process:
1. Train baseline model and get feature importances
2. Test removing each feature individually
3. Identify features that hurt accuracy when removed
4. Remove all features that don't hurt (or help) when removed
5. Retrain with remaining features
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("remove_negative")


def train_and_get_accuracy(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, xgb_params: dict, scale_pos_weight: float) -> float:
    """Train model and return accuracy."""
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
    return accuracy_score(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Remove features that hurt accuracy")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/remove_negative_features_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("REMOVING NEGATIVE FEATURES")
    log.info("="*80)
    
    # Build dataset
    log.info(f"\nLoading dataset from {image_dir}...")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    dataset_start = time.perf_counter()
    X, y, filenames = build_dataset(image_dir, repo=repo)
    dataset_time = time.perf_counter() - dataset_start
    
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset loaded in {dataset_time:.2f}s: {len(y)} samples, {X.shape[1]} features")
    
    # Split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Setup
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Baseline
    log.info("\n" + "="*80)
    log.info("STEP 1: Baseline model")
    log.info("="*80)
    
    baseline_start = time.perf_counter()
    baseline_acc = train_and_get_accuracy(X_train, X_test, y_train, y_test, xgb_params, scale_pos_weight)
    baseline_time = time.perf_counter() - baseline_start
    
    log.info(f"Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    log.info(f"Training time: {baseline_time:.2f}s")
    
    # Get feature importances
    clf_baseline = Pipeline([
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
    clf_baseline.fit(X_train, y_train)
    importances = clf_baseline.named_steps["xgb"].feature_importances_
    
    # Sort features by importance
    feature_importance_pairs = [(i, importances[i]) for i in range(len(importances))]
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    log.info("\n" + "="*80)
    log.info("STEP 2: Feature importance ranking")
    log.info("="*80)
    log.info("\nAll features sorted by importance (most to least):")
    for rank, (feat_idx, imp) in enumerate(feature_importance_pairs, 1):
        log.info(f"  {rank:2d}. Feature {feat_idx:2d}: {imp:.6f}")
    
    # Test removing each feature
    log.info("\n" + "="*80)
    log.info("STEP 3: Testing feature removal (one at a time)")
    log.info("="*80)
    log.info("Testing each feature to see if removal hurts accuracy...")
    
    features_to_remove = []
    features_to_keep = []
    removal_results = []
    
    test_start = time.perf_counter()
    last_log_time = test_start
    
    for idx, (feat_idx, imp) in enumerate(feature_importance_pairs):
        current_time = time.perf_counter()
        if current_time - last_log_time >= 1.0 or idx % 10 == 0:
            log.info(f"Testing feature {idx+1}/{len(feature_importance_pairs)}: feature {feat_idx}")
            last_log_time = current_time
        
        # Test removing this feature
        features_without = [i for i in range(X.shape[1]) if i != feat_idx]
        X_train_test = X_train[:, features_without]
        X_test_test = X_test[:, features_without]
        
        test_acc = train_and_get_accuracy(X_train_test, X_test_test, y_train, y_test, xgb_params, scale_pos_weight)
        acc_change = test_acc - baseline_acc
        
        removal_results.append({
            "feature_idx": int(feat_idx),
            "importance": float(imp),
            "accuracy_without": float(test_acc),
            "accuracy_change": float(acc_change),
        })
        
        # If removing this feature doesn't hurt (or helps), mark for removal
        if test_acc >= baseline_acc - 0.0001:  # Allow tiny tolerance
            features_to_remove.append(feat_idx)
        else:
            features_to_keep.append(feat_idx)
    
    test_time = time.perf_counter() - test_start
    log.info(f"\nFeature removal testing completed in {test_time:.2f}s")
    
    # Summary
    log.info("\n" + "="*80)
    log.info("STEP 4: Results")
    log.info("="*80)
    
    log.info(f"\nFeatures to REMOVE ({len(features_to_remove)}):")
    for feat_idx in sorted(features_to_remove):
        result = next(r for r in removal_results if r["feature_idx"] == feat_idx)
        log.info(f"  Feature {feat_idx:2d}: removal improves/maintains accuracy ({result['accuracy_change']*100:+.2f}%)")
    
    log.info(f"\nFeatures to KEEP ({len(features_to_keep)}):")
    for feat_idx in sorted(features_to_keep):
        result = next(r for r in removal_results if r["feature_idx"] == feat_idx)
        log.info(f"  Feature {feat_idx:2d}: removal hurts accuracy ({result['accuracy_change']*100:+.2f}%)")
    
    # Train final model with only beneficial features
    if features_to_remove:
        log.info("\n" + "="*80)
        log.info("STEP 5: Final model with beneficial features only")
        log.info("="*80)
        
        X_train_final = X_train[:, features_to_keep]
        X_test_final = X_test[:, features_to_keep]
        
        final_acc = train_and_get_accuracy(X_train_final, X_test_final, y_train, y_test, xgb_params, scale_pos_weight)
        final_f1 = f1_score(y_test, Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight, objective="binary:logistic", eval_metric="logloss", **xgb_params)),
        ]).fit(X_train_final, y_train).predict(X_test_final))
        
        improvement = final_acc - baseline_acc
        
        log.info(f"\nFinal Results:")
        log.info(f"  Baseline: {baseline_acc:.4f} ({X.shape[1]} features)")
        log.info(f"  Final: {final_acc:.4f} ({len(features_to_keep)} features)")
        log.info(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        log.info(f"  F1: {final_f1:.4f}")
        log.info(f"  Features removed: {len(features_to_remove)}")
    else:
        log.info("\nNo features can be safely removed - all features contribute positively")
        final_acc = baseline_acc
        improvement = 0.0
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline_accuracy": float(baseline_acc),
            "baseline_n_features": X.shape[1],
            "final_accuracy": float(final_acc),
            "final_n_features": len(features_to_keep),
            "improvement": float(improvement),
            "features_removed": features_to_remove,
            "features_kept": features_to_keep,
            "removal_results": removal_results,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

