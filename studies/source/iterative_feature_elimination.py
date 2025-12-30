#!/usr/bin/env python3
"""Iteratively remove features that hurt accuracy, keeping only beneficial ones.

Process:
1. Get feature importances from baseline model
2. Remove least important features one by one
3. Retrain and test accuracy
4. Keep only features that improve or maintain accuracy
5. Repeat until all remaining features are beneficial
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
log = logging.getLogger("feat_elim")


def train_and_get_accuracy(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, xgb_params: dict, scale_pos_weight: float) -> tuple[float, xgb.XGBClassifier]:
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
    acc = accuracy_score(y_test, y_pred)
    
    return acc, clf.named_steps["xgb"]


def main():
    parser = argparse.ArgumentParser(description="Iterative feature elimination")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/iterative_elimination_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("ITERATIVE FEATURE ELIMINATION")
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
    log.info(f"  Keep: {np.sum(y == 1)}, Trash: {np.sum(y == 0)}")
    
    if len(y) < 20:
        log.error("Insufficient labeled data")
        return 1
    
    # Fixed split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Setup
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Step 1: Train baseline and get feature importances
    log.info("\n" + "="*80)
    log.info("STEP 1: Baseline model and feature importances")
    log.info("="*80)
    
    baseline_acc, baseline_model = train_and_get_accuracy(X_train, X_test, y_train, y_test, xgb_params, scale_pos_weight)
    log.info(f"Baseline accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    
    importances = baseline_model.feature_importances_
    feature_indices = np.arange(len(importances))
    
    # Sort by importance (ascending - least important first)
    sorted_indices = np.argsort(importances)
    
    log.info("\nAll features sorted by importance (least to most):")
    for i, idx in enumerate(sorted_indices):
        log.info(f"  Feature {idx:2d}: importance={importances[idx]:.6f}")
    
    # Step 2: Iterative elimination
    log.info("\n" + "="*80)
    log.info("STEP 2: Iterative feature elimination")
    log.info("="*80)
    
    current_features = feature_indices.copy()
    best_acc = baseline_acc
    best_features = current_features.copy()
    elimination_history = []
    
    iteration = 0
    max_iterations = len(feature_indices) - 5  # Keep at least 5 features
    
    log.info(f"\nStarting with {len(current_features)} features")
    log.info(f"Target: Remove features that hurt accuracy")
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    while len(current_features) > 5 and iteration < max_iterations:
        iteration += 1
        log.info(f"\n--- Iteration {iteration} ---")
        log.info(f"Current features: {len(current_features)}")
        log.info(f"Current best accuracy: {best_acc:.4f}")
        
        # Get importances for current feature set (always retrain to get fresh importances)
        X_train_current = X_train[:, current_features]
        X_test_current = X_test[:, current_features]
        _, current_model = train_and_get_accuracy(X_train_current, X_test_current, y_train, y_test, xgb_params, scale_pos_weight)
        current_importances = current_model.feature_importances_
        
        # Find least important features in current set
        feature_importance_pairs = [(current_features[i], current_importances[i]) for i in range(len(current_features))]
        feature_importance_pairs.sort(key=lambda x: x[1])  # Sort by importance (ascending)
        
        # Get current accuracy with all current features
        X_train_current = X_train[:, current_features]
        X_test_current = X_test[:, current_features]
        current_acc, _ = train_and_get_accuracy(X_train_current, X_test_current, y_train, y_test, xgb_params, scale_pos_weight)
        
        # Try removing the bottom 5% of features (or at least 1)
        n_to_remove = max(1, int(len(current_features) * 0.05))
        features_to_remove = [feature_importance_pairs[i][0] for i in range(n_to_remove)]
        
        log.info(f"Testing removal of {n_to_remove} least important features: {features_to_remove}")
        for idx in features_to_remove:
            imp = current_importances[np.where(current_features == idx)[0][0]]
            log.info(f"  Feature {idx}: importance={imp:.6f}")
        
        # Test removing these features
        features_to_test = [f for f in current_features if f not in features_to_remove]
        X_train_test = X_train[:, features_to_test]
        X_test_test = X_test[:, features_to_test]
        
        test_acc, _ = train_and_get_accuracy(X_train_test, X_test_test, y_train, y_test, xgb_params, scale_pos_weight)
        acc_change = test_acc - current_acc
        
        log.info(f"  Current accuracy (with features): {current_acc:.4f}")
        log.info(f"  Accuracy with features removed: {test_acc:.4f} ({acc_change:+.4f}, {acc_change*100:+.2f}%)")
        
        # More lenient threshold: allow removal if accuracy drop is < 1%
        threshold = current_acc - 0.01  # Allow 1% drop to find truly useless features
        
        if test_acc >= threshold:
            # Removing these features helps or doesn't hurt significantly
            improvement = "improved" if test_acc > current_acc else ("maintained" if test_acc >= current_acc - 0.0001 else f"slight drop ({acc_change*100:.2f}%)")
            log.info(f"  ✅ Removing {len(features_to_remove)} features - accuracy {improvement}")
            current_features = np.array(features_to_test)
            if test_acc > best_acc:
                best_acc = test_acc
                best_features = current_features.copy()
            
            for removed_idx in features_to_remove:
                imp = current_importances[np.where(current_features == removed_idx)[0][0]] if removed_idx in current_features else 0.0
                elimination_history.append({
                    "iteration": iteration,
                    "removed_feature": int(removed_idx),
                    "feature_importance": float(imp),
                    "accuracy_before": float(current_acc),
                    "accuracy_after": float(test_acc),
                    "accuracy_change": float(acc_change),
                    "n_features_remaining": len(current_features),
                })
        else:
            # Removing these features hurts - keep them
            log.info(f"  ❌ Keeping {len(features_to_remove)} features - removal hurts accuracy")
            # Try removing fewer features (just the single least important)
            if len(feature_importance_pairs) > 0:
                least_important_idx = feature_importance_pairs[0][0]
                least_important_imp = feature_importance_pairs[0][1]
                log.info(f"  Trying single feature removal: feature {least_important_idx}")
                features_to_test_single = [f for f in current_features if f != least_important_idx]
                X_train_test_single = X_train[:, features_to_test_single]
                X_test_test_single = X_test[:, features_to_test_single]
                test_acc_single, _ = train_and_get_accuracy(X_train_test_single, X_test_test_single, y_train, y_test, xgb_params, scale_pos_weight)
                acc_change_single = test_acc_single - current_acc
                log.info(f"    Single removal accuracy: {test_acc_single:.4f} ({acc_change_single:+.4f}, {acc_change_single*100:+.2f}%)")
                if test_acc_single >= threshold:
                    log.info(f"  ✅ Removing single feature {least_important_idx}")
                    current_features = np.array(features_to_test_single)
                    if test_acc_single > best_acc:
                        best_acc = test_acc_single
                        best_features = current_features.copy()
                    elimination_history.append({
                        "iteration": iteration,
                        "removed_feature": int(least_important_idx),
                        "feature_importance": float(least_important_imp),
                        "accuracy_before": float(current_acc),
                        "accuracy_after": float(test_acc_single),
                        "accuracy_change": float(acc_change_single),
                        "n_features_remaining": len(current_features),
                    })
                else:
                    log.info(f"  ⚠️  No features can be safely removed")
                    break
            else:
                log.info(f"  ⚠️  No more features to test")
                break
        
        # Progress update
        if iteration % 5 == 0:
            log.info(f"\nProgress: {len(current_features)} features remaining, best accuracy: {best_acc:.4f}")
    
    # Final model with best features
    log.info("\n" + "="*80)
    log.info("FINAL RESULTS")
    log.info("="*80)
    
    log.info(f"\nBaseline: {baseline_acc:.4f} ({len(feature_indices)} features)")
    log.info(f"Final: {best_acc:.4f} ({len(best_features)} features)")
    log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({(best_acc - baseline_acc)*100:+.2f}%)")
    log.info(f"Features removed: {len(feature_indices) - len(best_features)}")
    
    # Train final model and get metrics
    X_train_final = X_train[:, best_features]
    X_test_final = X_test[:, best_features]
    final_acc, final_model = train_and_get_accuracy(X_train_final, X_test_final, y_train, y_test, xgb_params, scale_pos_weight)
    
    y_pred_final = final_model.predict(X_test_final)
    final_f1 = f1_score(y_test, y_pred_final)
    
    log.info(f"\nFinal model metrics:")
    log.info(f"  Accuracy: {final_acc:.4f}")
    log.info(f"  F1: {final_f1:.4f}")
    
    # Feature importance ranking for final model
    final_importances = final_model.feature_importances_
    final_feature_importance_pairs = [(best_features[i], final_importances[i]) for i in range(len(best_features))]
    final_feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    log.info(f"\nFinal feature importance ranking:")
    for rank, (feat_idx, imp) in enumerate(final_feature_importance_pairs, 1):
        log.info(f"  {rank:2d}. Feature {feat_idx:2d}: {imp:.6f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline_accuracy": float(baseline_acc),
            "baseline_n_features": len(feature_indices),
            "final_accuracy": float(final_acc),
            "final_n_features": len(best_features),
            "improvement": float(final_acc - baseline_acc),
            "features_removed": int(len(feature_indices) - len(best_features)),
            "final_features": [int(f) for f in sorted(best_features)],
            "final_feature_importances": {int(feat_idx): float(imp) for feat_idx, imp in final_feature_importance_pairs},
            "elimination_history": elimination_history,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

