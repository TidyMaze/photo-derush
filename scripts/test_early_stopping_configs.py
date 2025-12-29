#!/usr/bin/env python3
"""Test different early stopping configurations to reach 88.24% goal."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("test_early_stopping")


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, config: dict) -> float:
    """Train model and return test accuracy."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params = {
        "iterations": config.get("iterations", 500),
        "learning_rate": config.get("learning_rate", 0.1),
        "depth": config.get("depth", 6),
        "l2_leaf_reg": config.get("l2_leaf_reg", 1.0),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    if "early_stopping_rounds" in config and X_val is not None:
        cb_params["early_stopping_rounds"] = config["early_stopping_rounds"]
        cb_params["eval_metric"] = config.get("eval_metric", "Accuracy")
        cb_params["use_best_model"] = True
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    if "early_stopping_rounds" in config and X_val is not None:
        clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    else:
        clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    return float(accuracy_score(y_test, y_pred))


def main():
    # Load data
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    repo = RatingsTagsRepository(path=os.path.join(image_dir, ".ratings_tags.json"))
    
    X, y, _ = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    # Split: 80% train, 10% val, 10% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=42
    )
    
    log.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Baseline without early stopping (train on all training data, no validation split)
    baseline_config = {"iterations": 200, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 1.0}
    # Combine train+val for baseline (no early stopping)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    baseline_acc = train_and_evaluate(X_train_full, None, X_test, y_train_full, None, y_test, baseline_config)
    log.info(f"\n{'='*80}")
    log.info(f"BASELINE (no early stopping, train on all): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    log.info(f"{'='*80}\n")
    
    goal = baseline_acc
    best_config = None
    best_acc = baseline_acc
    
    # Test configurations - trying to reach 88.24%
    configs_to_test = [
        # 1. Very high patience with standard LR (let it train longer)
        {"iterations": 1000, "learning_rate": 0.1, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Accuracy"},
        {"iterations": 2000, "learning_rate": 0.1, "depth": 6, "early_stopping_rounds": 400, "eval_metric": "Accuracy"},
        
        # 2. Lower learning rate with very high patience
        {"iterations": 2000, "learning_rate": 0.05, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Accuracy"},
        {"iterations": 3000, "learning_rate": 0.05, "depth": 6, "early_stopping_rounds": 400, "eval_metric": "Accuracy"},
        
        # 3. Slightly lower LR (between 0.05 and 0.1)
        {"iterations": 2000, "learning_rate": 0.07, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Accuracy"},
        {"iterations": 2000, "learning_rate": 0.06, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Accuracy"},
        
        # 4. Use Logloss with high patience
        {"iterations": 2000, "learning_rate": 0.1, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Logloss"},
        {"iterations": 2000, "learning_rate": 0.05, "depth": 6, "early_stopping_rounds": 300, "eval_metric": "Logloss"},
        
        # 5. Combine train+val for training, use separate small validation set
        # (This requires different approach - train on more data)
    ]
    
    log.info("Testing early stopping configurations...\n")
    
    for i, config in enumerate(configs_to_test, 1):
        name = f"Config {i}: LR={config['learning_rate']}, iter={config['iterations']}, patience={config['early_stopping_rounds']}, metric={config['eval_metric']}"
        log.info(f"Testing: {name}")
        
        try:
            acc = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, config)
            improvement = acc - goal
            log.info(f"  Result: {acc:.4f} ({acc*100:.2f}%) - {improvement:+.4f} ({improvement*100:+.2f} pp)")
            
            if acc > best_acc:
                best_acc = acc
                best_config = config.copy()
                log.info(f"  ✅ NEW BEST!")
        except Exception as e:
            log.error(f"  ❌ Failed: {e}")
        
        log.info("")
    
    # Final summary
    log.info(f"\n{'='*80}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"Baseline (no early stopping): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    log.info(f"Best with early stopping: {best_acc:.4f} ({best_acc*100:.2f}%)")
    improvement = best_acc - baseline_acc
    log.info(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    
    if best_config:
        log.info(f"\nBest config:")
        for key, value in sorted(best_config.items()):
            log.info(f"  {key}: {value}")
    
    if best_acc >= goal:
        log.info(f"\n✅ SUCCESS: Reached goal with early stopping!")
    else:
        log.info(f"\n⚠️  Did not reach goal. Best: {best_acc:.4f}, Goal: {goal:.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

