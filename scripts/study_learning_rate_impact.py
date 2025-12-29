#!/usr/bin/env python3
"""Study the impact of learning rate on validation accuracy.

Tests different learning rates and analyzes:
- Final validation accuracy
- Training convergence speed
- Overfitting behavior
- Best iteration found
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("study_lr")


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, learning_rate: float, max_iterations: int = 2000, patience: int = 200) -> dict:
    """Train model with given learning rate and return detailed metrics."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params = {
        "iterations": max_iterations,
        "learning_rate": learning_rate,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": patience,
        "eval_metric": "Accuracy",
        "use_best_model": True,
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    # Get model details
    cat_model = clf.named_steps["cat"]
    total_iterations = cat_model.tree_count_
    best_iteration = cat_model.best_iteration_ if hasattr(cat_model, "best_iteration_") else total_iterations - 1
    
    # Get training history if available
    training_history = {}
    if hasattr(cat_model, "evals_result_"):
        evals = cat_model.evals_result_
        if evals:
            for key in evals:
                if "test" in key.lower() or "validation" in key.lower():
                    if "Accuracy" in evals[key]:
                        training_history["val_accuracy"] = evals[key]["Accuracy"]
                    if "Logloss" in evals[key]:
                        training_history["val_logloss"] = evals[key]["Logloss"]
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    return {
        "learning_rate": learning_rate,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "total_iterations": total_iterations,
        "best_iteration": best_iteration,
        "early_stopped": best_iteration < total_iterations - 1,
        "training_history": training_history,
    }


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
    
    # Test different learning rates
    learning_rates = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.12, 0.15, 0.2, 0.25, 0.3
    ]
    
    log.info("\n" + "="*80)
    log.info("STUDYING LEARNING RATE IMPACT")
    log.info("="*80)
    log.info(f"Testing {len(learning_rates)} learning rates: {learning_rates}\n")
    
    results = []
    
    for lr in learning_rates:
        log.info(f"Testing learning_rate={lr:.3f}...")
        try:
            result = train_and_evaluate(
                X_train, X_val, X_test, y_train, y_val, y_test,
                learning_rate=lr, max_iterations=2000, patience=200
            )
            results.append(result)
            
            log.info(f"  Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
            log.info(f"  Best iteration: {result['best_iteration']} (total: {result['total_iterations']})")
            log.info(f"  Early stopped: {result['early_stopped']}")
            log.info("")
        except Exception as e:
            log.error(f"  Failed: {e}")
            log.info("")
    
    # Analysis
    log.info("\n" + "="*80)
    log.info("ANALYSIS")
    log.info("="*80)
    
    if not results:
        log.error("No results to analyze")
        return 1
    
    # Find best learning rate
    best_result = max(results, key=lambda r: r["test_accuracy"])
    worst_result = min(results, key=lambda r: r["test_accuracy"])
    
    log.info(f"\nBest Learning Rate: {best_result['learning_rate']:.3f}")
    log.info(f"  Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    log.info(f"  F1: {best_result['test_f1']:.4f}")
    log.info(f"  ROC-AUC: {best_result['test_roc_auc']:.4f}")
    log.info(f"  Best iteration: {best_result['best_iteration']}")
    log.info(f"  Total iterations: {best_result['total_iterations']}")
    
    log.info(f"\nWorst Learning Rate: {worst_result['learning_rate']:.3f}")
    log.info(f"  Test Accuracy: {worst_result['test_accuracy']:.4f} ({worst_result['test_accuracy']*100:.2f}%)")
    
    # Summary table
    log.info("\n" + "="*80)
    log.info("SUMMARY TABLE")
    log.info("="*80)
    log.info(f"{'LR':<8} {'Accuracy':<10} {'F1':<8} {'ROC-AUC':<10} {'Best Iter':<10} {'Early Stop':<12}")
    log.info("-" * 80)
    
    for r in sorted(results, key=lambda x: x["learning_rate"]):
        early_stop_str = "Yes" if r["early_stopped"] else "No"
        log.info(
            f"{r['learning_rate']:<8.3f} {r['test_accuracy']:<10.4f} {r['test_f1']:<8.4f} "
            f"{r['test_roc_auc']:<10.4f} {r['best_iteration']:<10} {early_stop_str:<12}"
        )
    
    # Find optimal range
    accuracies = [r["test_accuracy"] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    log.info(f"\nStatistics:")
    log.info(f"  Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    log.info(f"  Std accuracy: {std_acc:.4f} ({std_acc*100:.2f}%)")
    log.info(f"  Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
    
    # Find learning rates within 1% of best
    threshold = best_result["test_accuracy"] - 0.01
    good_lrs = [r for r in results if r["test_accuracy"] >= threshold]
    log.info(f"\nLearning rates within 1% of best ({best_result['test_accuracy']:.4f}):")
    for r in sorted(good_lrs, key=lambda x: x["learning_rate"]):
        log.info(f"  LR={r['learning_rate']:.3f}: {r['test_accuracy']:.4f} ({r['test_accuracy']*100:.2f}%)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

