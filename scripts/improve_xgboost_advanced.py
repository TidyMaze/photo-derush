#!/usr/bin/env python3
"""Advanced strategies to improve XGBoost accuracy.

New strategies:
1. Alternative algorithms (LightGBM, CatBoost)
2. Stacking ensemble (meta-learner)
3. Threshold optimization (optimize decision threshold)
4. More aggressive hyperparameter tuning
5. Feature engineering (target encoding, binning)
6. Cross-validation with more folds

Usage:
    poetry run python scripts/improve_xgboost_advanced.py [IMAGE_DIR]
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("improve_xgb")


def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_name: str, xgb_params_override: dict | None = None) -> dict:
    """Train and evaluate baseline XGBoost model."""
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    if xgb_params_override:
        xgb_params.update(xgb_params_override)
    
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
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "model": clf,
        "y_proba": y_proba,
    }


def test_lightgbm(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test LightGBM as alternative to XGBoost."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 1: LightGBM")
    log.info("="*80)
    
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM not available")
        return {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # LightGBM with similar params to XGBoost
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": xgb_params.get("max_depth", 6) * 2,
        "learning_rate": xgb_params.get("learning_rate", 0.1),
        "feature_fraction": xgb_params.get("colsample_bytree", 1.0),
        "bagging_fraction": xgb_params.get("subsample", 1.0),
        "bagging_freq": 5,
        "min_child_samples": xgb_params.get("min_child_weight", 1),
        "reg_alpha": xgb_params.get("reg_alpha", 0),
        "reg_lambda": xgb_params.get("reg_lambda", 1),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    return {
        "metrics": {
            "accuracy": acc,
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": acc_diff,
    }


def test_catboost(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test CatBoost as alternative to XGBoost."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 2: CatBoost")
    log.info("="*80)
    
    try:
        import catboost as cb
    except ImportError:
        log.warning("CatBoost not available")
        return {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    cb_params = {
        "iterations": xgb_params.get("n_estimators", 100),
        "learning_rate": xgb_params.get("learning_rate", 0.1),
        "depth": xgb_params.get("max_depth", 6),
        "l2_leaf_reg": xgb_params.get("reg_lambda", 1),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    clf = cb.CatBoostClassifier(**cb_params)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    return {
        "metrics": {
            "accuracy": acc,
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": acc_diff,
    }


def test_threshold_optimization(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Optimize decision threshold instead of using default 0.5."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 3: Threshold Optimization")
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
    
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold using validation set
    # Use cross-validation to find threshold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_acc = 0.0
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        clf_cv = Pipeline([
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
        clf_cv.fit(X_cv_train, y_cv_train)
        y_cv_proba = clf_cv.predict_proba(X_cv_val)[:, 1]
        
        for threshold in thresholds:
            y_cv_pred = (y_cv_proba >= threshold).astype(int)
            acc = accuracy_score(y_cv_val, y_cv_pred)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
    
    log.info(f"  Optimal threshold: {best_threshold:.3f} (default: 0.5)")
    
    # Apply optimal threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    return {
        "metrics": {
            "accuracy": acc,
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": acc_diff,
        "optimal_threshold": float(best_threshold),
    }


def test_stacking_ensemble(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test stacking ensemble with meta-learner."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 4: Stacking Ensemble")
    log.info("="*80)
    
    import xgboost as xgb
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Base models
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        **xgb_params,
    )
    
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            num_leaves=31,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        base_models = [("xgb", xgb_clf), ("lgb", lgb_clf)]
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        base_models = [("xgb", xgb_clf), ("rf", rf_clf)]
    
    # Meta-learner
    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    
    # Stacking
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1,
    )
    
    stacking_clf.fit(X_train_scaled, y_train)
    y_pred = stacking_clf.predict(X_test_scaled)
    y_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    return {
        "metrics": {
            "accuracy": acc,
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": acc_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced XGBoost improvement strategies")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/xgboost_advanced_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("ADVANCED XGBOOST IMPROVEMENT STRATEGIES")
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
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Baseline
    log.info("\n" + "="*80)
    log.info("BASELINE: XGBoost")
    log.info("="*80)
    
    baseline_start = time.perf_counter()
    baseline_result = train_and_evaluate(X_train, X_test, y_train, y_test, "Baseline")
    baseline_time = time.perf_counter() - baseline_start
    baseline_acc = baseline_result["metrics"]["accuracy"]
    
    log.info(f"\nBaseline Results (trained in {baseline_time:.2f}s):")
    log.info(f"  Accuracy: {baseline_acc:.4f}")
    log.info(f"  F1: {baseline_result['metrics']['f1']:.4f}")
    log.info(f"  ROC-AUC: {baseline_result['metrics']['roc_auc']:.4f}")
    
    # Test strategies
    all_results = {
        "baseline": baseline_result["metrics"],
        "baseline_time": float(baseline_time),
        "strategies": {},
    }
    
    # Strategy 1: LightGBM
    try:
        lgb_results = test_lightgbm(X_train, X_test, y_train, y_test, baseline_acc)
        if lgb_results:
            all_results["strategies"]["lightgbm"] = lgb_results
    except Exception as e:
        log.error(f"LightGBM failed: {e}")
    
    # Strategy 2: CatBoost
    try:
        cb_results = test_catboost(X_train, X_test, y_train, y_test, baseline_acc)
        if cb_results:
            all_results["strategies"]["catboost"] = cb_results
    except Exception as e:
        log.error(f"CatBoost failed: {e}")
    
    # Strategy 3: Threshold Optimization
    try:
        threshold_results = test_threshold_optimization(X_train, X_test, y_train, y_test, baseline_acc)
        if threshold_results:
            all_results["strategies"]["threshold_optimization"] = threshold_results
    except Exception as e:
        log.error(f"Threshold optimization failed: {e}")
    
    # Strategy 4: Stacking Ensemble
    try:
        stacking_results = test_stacking_ensemble(X_train, X_test, y_train, y_test, baseline_acc)
        if stacking_results:
            all_results["strategies"]["stacking"] = stacking_results
    except Exception as e:
        log.error(f"Stacking failed: {e}")
    
    # Summary
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    log.info(f"\nBaseline: {baseline_acc:.4f}")
    
    best_strategy = None
    best_acc = baseline_acc
    
    for strategy_name, results in all_results["strategies"].items():
        if "metrics" in results:
            acc = results["metrics"]["accuracy"]
            diff = results.get("accuracy_diff", acc - baseline_acc)
            log.info(f"{strategy_name}: {acc:.4f} ({diff:+.4f}, {diff*100:+.2f}%)")
            if acc > best_acc:
                best_acc = acc
                best_strategy = strategy_name
    
    if best_strategy:
        log.info(f"\n✅ Best strategy: {best_strategy} ({best_acc:.4f}, +{(best_acc-baseline_acc)*100:.2f}%)")
    else:
        log.info(f"\n⚠️  No strategy outperformed baseline")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

