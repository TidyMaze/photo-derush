#!/usr/bin/env python3
"""Quick training and evaluation with person detection feature.

Optimized for speed:
- Reduced n_estimators (100 instead of 200)
- Skips cross-validation
- Minimal logging
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.training_core import train_keep_trash_model

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quick train and evaluate")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/Pictures/photo-dataset"), help="Image directory")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators (default: 100)")
    args = parser.parse_args()
    
    image_dir = os.path.expanduser(args.image_dir)
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    print("="*80)
    print("QUICK TRAINING & EVALUATION")
    print("="*80)
    
    # Build dataset
    print(f"\nLoading dataset from {image_dir}...")
    t0 = time.perf_counter()
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        print("❌ No repository found")
        return 1
    
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    t1 = time.perf_counter()
    
    print(f"Dataset loaded in {(t1-t0):.2f}s: {len(y)} samples")
    print(f"  Keep: {np.sum(y == 1)}, Trash: {np.sum(y == 0)}")
    print(f"  Features: {FEATURE_COUNT}")
    
    if len(y) < 10:
        print("❌ Insufficient data")
        return 1
    
    # Split
    print(f"\nSplitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Train
    print(f"\nTraining model (n_estimators={args.n_estimators})...")
    t0 = time.perf_counter()
    temp_model_path = os.path.join(image_dir, ".quick_train_temp.joblib")
    
    # Build filenames for training set
    train_indices = []
    test_indices = []
    train_idx = 0
    test_idx = 0
    for i, fname in enumerate(filenames):
        if train_idx < len(y_train) and np.array_equal(y[i], y_train[train_idx]):
            train_indices.append(i)
            train_idx += 1
        else:
            test_indices.append(i)
            test_idx += 1
    
    train_filenames = [filenames[i] for i in train_indices[:len(y_train)]]
    
    result = train_keep_trash_model(
        image_dir=image_dir,
        model_path=temp_model_path,
        repo=repo,
        displayed_filenames=train_filenames,
        n_estimators=args.n_estimators,
        random_state=42,
        early_stopping_rounds=None,  # Skip early stopping for speed
    )
    t1 = time.perf_counter()
    
    if result is None:
        print("❌ Training failed")
        return 1
    
    print(f"Training completed in {(t1-t0):.2f}s")
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    import joblib
    data = joblib.load(temp_model_path)
    clf = data["model"]
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Feature importances
    if result.feature_importances:
        print(f"\nTop 10 Features:")
        for idx, (feat_idx, importance) in enumerate(result.feature_importances[:10], 1):
            print(f"  {idx}. Feature {feat_idx}: {importance:.6f}")
    
    # Cleanup
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    print("\n" + "="*80)
    print("✅ Training and evaluation complete")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

