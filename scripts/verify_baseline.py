#!/usr/bin/env python3
"""Verify baseline XGBoost performance can be reproduced."""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
log = logging.getLogger("verify")


def main():
    parser = argparse.ArgumentParser(description="Verify baseline performance")
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
    log.info("VERIFYING BASELINE PERFORMANCE")
    log.info("="*80)
    
    # Build dataset
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
    log.info(f"  Keep: {np.sum(y == 1)}, Trash: {np.sum(y == 0)}")
    
    # Fixed split (same as used elsewhere)
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load best params
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    log.info(f"\nLoaded hyperparameters: {xgb_params}")
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"Scale pos weight: {scale_pos_weight:.4f}")
    
    # Train model
    log.info("\nTraining baseline model...")
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
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    log.info(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    log.info(f"F1: {f1:.4f}")
    
    expected_acc = 0.7901
    diff = abs(acc - expected_acc)
    
    log.info(f"\nExpected baseline: {expected_acc:.4f}")
    log.info(f"Reproduced: {acc:.4f}")
    log.info(f"Difference: {diff:.4f}")
    
    if diff < 0.001:
        log.info("✅ Baseline successfully reproduced!")
        return 0
    elif diff < 0.01:
        log.info("⚠️  Close match (within 1%)")
        return 0
    else:
        log.warning("❌ Significant difference - may indicate non-reproducibility")
        return 1


if __name__ == "__main__":
    sys.exit(main())

