#!/usr/bin/env python3
"""Evaluate current model and test improvements for validation accuracy.

Compares:
1. Current model (baseline)
2. Improved hyperparameters (AVA-learned)
3. With feature interactions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.repository import RatingsTagsRepository
from src.training_core import train_keep_trash_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("eval_improvements")


def evaluate_model(clf: Pipeline, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate model and return metrics."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    
    log.info(f"\n{name} Results:")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall: {metrics['recall']:.4f}")
    log.info(f"  F1: {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        log.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train baseline model (current production config)."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params = {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    log.info("Training baseline model (current production config)...")
    clf.fit(X_train, y_train)
    
    return clf


def train_improved_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train improved model with moderate regularization (adapted for dataset size)."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    n_samples = len(X_train)
    
    # Adapt hyperparameters based on dataset size
    if n_samples < 500:
        # Small dataset: more iterations, moderate learning rate, less aggressive early stopping
        cb_params = {
            "iterations": 1000,  # More iterations
            "learning_rate": 0.03,  # Moderate LR
            "depth": 6,
            "l2_leaf_reg": 1.5,  # Moderate regularization
            "rsm": 0.95,  # Less feature dropout
            "bootstrap_type": "Bernoulli",
            "subsample": 0.95,  # Less row dropout
            "scale_pos_weight": scale_pos_weight,
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "early_stopping_rounds": 100,  # More patience
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }
        log.info("Using small-dataset-adapted hyperparameters (more iterations, more patience)")
    else:
        # Larger dataset: use AVA-learned hyperparameters
        cb_params = {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "rsm": 0.88,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "scale_pos_weight": scale_pos_weight,
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "early_stopping_rounds": 200,
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }
        log.info("Using AVA-learned hyperparameters")
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    log.info("Training improved model (adapted hyperparameters)...")
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    return clf


def main():
    parser = argparse.ArgumentParser(description="Evaluate model improvements")
    parser.add_argument("--image-dir", default=None, help="Image directory (default: from app config)")
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        # Try to get from app config or use default
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("EVALUATING MODEL IMPROVEMENTS")
    log.info("="*80)
    log.info(f"Image directory: {image_dir}")
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info("Building dataset...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(y)
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    
    log.info(f"Dataset: {n_samples} samples ({n_keep} keep, {n_trash} trash)")
    
    if n_samples < 20:
        log.error("Insufficient labeled data")
        return 1
    
    # Split: 80% train, 10% val, 10% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=42  # 0.125 * 0.8 = 0.1
    )
    
    log.info(f"\nData splits:")
    log.info(f"  Train: {len(X_train)} samples")
    log.info(f"  Val: {len(X_val)} samples")
    log.info(f"  Test: {len(X_test)} samples")
    
    # Train baseline model
    log.info("\n" + "="*80)
    log.info("BASELINE MODEL (Current Production Config)")
    log.info("="*80)
    baseline_clf = train_baseline_model(X_train, X_test, y_train, y_test)
    baseline_metrics = evaluate_model(baseline_clf, X_test, y_test, "Baseline")
    
    # Train improved model
    log.info("\n" + "="*80)
    log.info("IMPROVED MODEL (AVA-Learned Hyperparameters)")
    log.info("="*80)
    improved_clf = train_improved_model(X_train, X_val, X_test, y_train, y_val, y_test)
    improved_metrics = evaluate_model(improved_clf, X_test, y_test, "Improved")
    
    # Compare
    log.info("\n" + "="*80)
    log.info("COMPARISON")
    log.info("="*80)
    improvement = improved_metrics["accuracy"] - baseline_metrics["accuracy"]
    log.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
    log.info(f"Improved Accuracy: {improved_metrics['accuracy']:.4f} ({improved_metrics['accuracy']*100:.2f}%)")
    log.info(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    
    if improvement > 0.01:  # > 1% improvement
        log.info("\n✅ Significant improvement detected! Consider updating production model.")
    elif improvement > 0:
        log.info("\n✅ Small improvement detected.")
    else:
        log.info("\n⚠️  No improvement. Current model may already be optimal for this dataset.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

