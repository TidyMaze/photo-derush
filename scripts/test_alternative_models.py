#!/usr/bin/env python3
"""Test alternative models and approaches to improve validation accuracy.

Tests:
1. Different tree-based models (LightGBM, CatBoost, Random Forest)
2. Neural networks (MLP)
3. Stacking/blending ensembles
4. Different feature engineering approaches

Usage:
    poetry run python scripts/test_alternative_models.py [IMAGE_DIR]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("alt_models")


def train_and_evaluate_model(clf, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Train and evaluate a model."""
    log.info(f"[{model_name}] Training...")
    start_time = time.perf_counter()
    
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    
    log.info(f"[{model_name}] Making predictions...")
    y_pred = clf.predict(X_test)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        # For SVM-like models, use decision function as proxy
        decision = clf.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-decision))  # Sigmoid transform
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    
    metrics["train_time"] = float(train_time)
    
    return metrics


def test_xgboost_baseline(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test XGBoost baseline."""
    log.info("\n" + "="*80)
    log.info("MODEL 1: XGBoost (Baseline)")
    log.info("="*80)
    
    import xgboost as xgb
    
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
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "XGBoost")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "XGBoost", "metrics": metrics}


def test_lightgbm(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test LightGBM."""
    log.info("\n" + "="*80)
    log.info("MODEL 2: LightGBM")
    log.info("="*80)
    
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM not available, skipping")
        return None
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lgb", lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
        )),
    ])
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "LightGBM")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "LightGBM", "metrics": metrics}


def test_catboost(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test CatBoost."""
    log.info("\n" + "="*80)
    log.info("MODEL 3: CatBoost")
    log.info("="*80)
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.warning("CatBoost not available, skipping")
        return None
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(
            random_state=42,
            thread_count=-1,
            scale_pos_weight=scale_pos_weight,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            iterations=200,
            learning_rate=0.1,
            depth=6,
        )),
    ])
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "CatBoost")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "CatBoost", "metrics": metrics}


def test_random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test Random Forest."""
    log.info("\n" + "="*80)
    log.info("MODEL 4: Random Forest")
    log.info("="*80)
    
    from sklearn.ensemble import RandomForestClassifier
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "RandomForest")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "RandomForest", "metrics": metrics}


def test_mlp(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test Multi-Layer Perceptron."""
    log.info("\n" + "="*80)
    log.info("MODEL 5: Neural Network (MLP)")
    log.info("="*80)
    
    from sklearn.neural_network import MLPClassifier
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "MLP")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "MLP", "metrics": metrics}


def test_stacking_ensemble(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Test stacking ensemble."""
    log.info("\n" + "="*80)
    log.info("MODEL 6: Stacking Ensemble")
    log.info("="*80)
    
    from sklearn.ensemble import StackingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    base_estimators = [
        ("xgb", Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                **xgb_params,
            )),
        ])),
        ("rf", Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])),
    ]
    
    clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=3,
    )
    
    metrics = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "Stacking")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return {"model": "Stacking", "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="Test alternative models")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/alternative_models_results.json", help="Output JSON file")
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
    log.info("TESTING ALTERNATIVE MODELS")
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
    
    # Test models
    results = []
    
    # Baseline
    baseline_result = test_xgboost_baseline(X_train, X_test, y_train, y_test)
    baseline_acc = baseline_result["metrics"]["accuracy"]
    results.append(baseline_result)
    
    # Alternative models
    model_tests = [
        test_lightgbm,
        test_catboost,
        test_random_forest,
        test_mlp,
        test_stacking_ensemble,
    ]
    
    for test_fn in model_tests:
        try:
            result = test_fn(X_train, X_test, y_train, y_test)
            if result:
                results.append(result)
                acc = result["metrics"]["accuracy"]
                acc_diff = acc - baseline_acc
                if acc_diff > 0.001:
                    log.info(f"  ✅ IMPROVEMENT: {acc_diff*100:+.2f}%")
        except Exception as e:
            log.error(f"Model test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    results_sorted = sorted(results, key=lambda x: x["metrics"]["accuracy"], reverse=True)
    
    log.info(f"\nModels ranked by accuracy:")
    log.info(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12} {'Time':<10} {'vs Baseline':<12}")
    log.info("-" * 80)
    
    for r in results_sorted:
        acc = r["metrics"]["accuracy"]
        f1 = r["metrics"]["f1"]
        roc = r["metrics"].get("roc_auc", float("nan"))
        train_t = r["metrics"]["train_time"]
        diff = acc - baseline_acc
        status = "✅" if diff > 0.001 else ("⚠️" if diff > -0.001 else "❌")
        log.info(f"{r['model']:<20} {acc:<12.4f} {f1:<12.4f} {roc:<12.4f} {train_t:<10.2f} {diff:+.4f} {status}")
    
    best_result = results_sorted[0]
    best_acc = best_result["metrics"]["accuracy"]
    best_diff = best_acc - baseline_acc
    
    log.info(f"\nBest Model: {best_result['model']}")
    log.info(f"  Accuracy: {best_acc:.4f} ({best_diff:+.4f}, {best_diff*100:+.2f}% vs baseline)")
    log.info(f"  F1: {best_result['metrics']['f1']:.4f}")
    log.info(f"  ROC-AUC: {best_result['metrics'].get('roc_auc', 'N/A')}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline_accuracy": float(baseline_acc),
            "best_model": best_result["model"],
            "best_accuracy": float(best_acc),
            "improvement": float(best_diff),
            "results": results,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

