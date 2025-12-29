#!/usr/bin/env python3
"""Simple, focused XGBoost hyperparameter tuning.

KISS approach: Just tune the most impactful parameters.
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("simple_tune")


def main():
    parser = argparse.ArgumentParser(description="Simple XGBoost tuning")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("SIMPLE XGBOOST HYPERPARAMETER TUNING")
    log.info("="*80)
    
    # Build dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    X, y, _ = build_dataset(image_dir, repo=repo)
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
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Baseline
    log.info("\nTraining baseline...")
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
        )),
    ])
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Simple grid search on key params
    log.info(f"\nTuning {args.trials} combinations...")
    
    best_acc = baseline_acc
    best_params = {}
    
    np.random.seed(42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i in range(args.trials):
        params = {
            "n_estimators": np.random.choice([100, 150, 200, 250, 300]),
            "learning_rate": np.random.choice([0.01, 0.05, 0.1, 0.15]),
            "max_depth": np.random.choice([3, 4, 5, 6, 7]),
            "min_child_weight": np.random.choice([1, 2, 3]),
            "subsample": np.random.choice([0.7, 0.8, 0.9, 1.0]),
            "colsample_bytree": np.random.choice([0.7, 0.8, 0.9, 1.0]),
            "reg_alpha": np.random.choice([0, 0.1, 0.5, 1.0]),
            "reg_lambda": np.random.choice([0.5, 1.0, 1.5, 2.0]),
        }
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                **params,
            )),
        ])
        
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        cv_acc = scores.mean()
        
        if cv_acc > best_acc:
            best_acc = cv_acc
            best_params = params.copy()
            log.info(f"Trial {i+1}/{args.trials}: {cv_acc:.4f} (improvement: {cv_acc-baseline_acc:+.4f})")
            log.info(f"  Params: {params}")
    
    # Test best params
    if best_params:
        log.info(f"\nTesting best params on test set...")
        best_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                **best_params,
            )),
        ])
        best_clf.fit(X_train, y_train)
        best_pred = best_clf.predict(X_test)
        best_test_acc = accuracy_score(y_test, best_pred)
        
        log.info(f"\nResults:")
        log.info(f"  Baseline: {baseline_acc:.4f}")
        log.info(f"  Best CV: {best_acc:.4f}")
        log.info(f"  Best test: {best_test_acc:.4f}")
        log.info(f"  Improvement: {best_test_acc - baseline_acc:+.4f}")
        log.info(f"\nBest params: {json.dumps(best_params, indent=2)}")
        
        # Save
        params_file = os.path.expanduser("~/.photo-derush-xgb-params.json")
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=2)
        log.info(f"Saved to {params_file}")
    else:
        log.info("No improvement found. Baseline is best.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

