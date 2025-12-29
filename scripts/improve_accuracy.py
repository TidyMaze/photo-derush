#!/usr/bin/env python3
"""Systematically improve model validation accuracy.

Strategies:
1. Feature selection (remove weak/redundant features)
2. Feature interactions (polynomial features)
3. Feature transformations (log, sqrt, etc.)
4. Ensemble methods
5. Hyperparameter optimization

Usage:
    poetry run python scripts/improve_accuracy.py [IMAGE_DIR]
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("improve_accuracy")


def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_name: str, xgb_params_override: dict | None = None) -> dict:
    """Train and evaluate a model."""
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
    }


def test_feature_selection(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test feature selection strategies."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 1: Feature Selection")
    log.info("="*80)
    
    results = {}
    
    # Test different numbers of top features
    n_features = X_train.shape[1]
    k_values = [n_features, int(n_features * 0.9), int(n_features * 0.8), int(n_features * 0.7), int(n_features * 0.6), int(n_features * 0.5)]
    k_values = sorted(set(k_values), reverse=True)
    
    for k in k_values:
        if k < 10:
            continue
        
        log.info(f"\nTesting SelectKBest with k={k} features...")
        
        # F-test based selection
        selector = SelectKBest(f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        result = train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, f"SelectKBest_f_{k}")
        acc = result["metrics"]["accuracy"]
        acc_diff = acc - baseline_acc
        
        log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
        
        results[f"kbest_f_{k}"] = {
            "k": k,
            "method": "f_classif",
            "metrics": result["metrics"],
            "accuracy_diff": float(acc_diff),
        }
        
        if acc_diff > 0.001:  # Found improvement
            log.info(f"  ✅ IMPROVEMENT FOUND!")
    
    return results


def test_feature_interactions(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test polynomial feature interactions."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 2: Feature Interactions (Polynomial)")
    log.info("="*80)
    
    results = {}
    
    # Test degree 2 interactions (selected features only to avoid explosion)
    log.info("\nTesting polynomial features (degree=2) on top 30 features...")
    
    # First select top 30 features
    selector = SelectKBest(f_classif, k=min(30, X_train.shape[1]))
    X_train_top = selector.fit_transform(X_train, y_train)
    X_test_top = selector.transform(X_test)
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_top)
    X_test_poly = poly.transform(X_test_top)
    
    log.info(f"  Original features: {X_train_top.shape[1]}, Polynomial features: {X_train_poly.shape[1]}")
    
    result = train_and_evaluate(X_train_poly, X_test_poly, y_train, y_test, "Polynomial_degree2")
    acc = result["metrics"]["accuracy"]
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    results["polynomial_degree2"] = {
        "metrics": result["metrics"],
        "accuracy_diff": float(acc_diff),
        "n_features": X_train_poly.shape[1],
    }
    
    return results


def test_feature_transformations(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test feature transformations (log, sqrt, etc.)."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 3: Feature Transformations")
    log.info("="*80)
    
    results = {}
    
    # Test log transformation on positive features
    log.info("\nTesting log(1+x) transformation on positive features...")
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    
    # Apply log to features that are always positive (like histograms, ratios)
    # Skip features that might be negative
    for i in range(X_train.shape[1]):
        if np.min(X_train[:, i]) >= 0:
            X_train_log[:, i] = np.log1p(X_train[:, i])
            X_test_log[:, i] = np.log1p(X_test[:, i])
    
    result = train_and_evaluate(X_train_log, X_test_log, y_train, y_test, "Log_Transform")
    acc = result["metrics"]["accuracy"]
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    results["log_transform"] = {
        "metrics": result["metrics"],
        "accuracy_diff": float(acc_diff),
    }
    
    return results


def test_calibration(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test probability calibration."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 4: Probability Calibration")
    log.info("="*80)
    
    results = {}
    
    from sklearn.calibration import CalibratedClassifierCV
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    base_clf = Pipeline([
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
    
    # Test isotonic calibration
    log.info("\nTesting Isotonic Calibration...")
    calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    calibrated_clf.fit(X_train, y_train)
    
    y_pred = calibrated_clf.predict(X_test)
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    results["isotonic_calibration"] = {
        "metrics": {
            "accuracy": acc,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": float(acc_diff),
    }
    
    return results


def test_optuna_tuning(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test Optuna-based hyperparameter optimization."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 3: Optuna Hyperparameter Optimization")
    log.info("="*80)
    
    results = {}
    
    try:
        import optuna
    except ImportError:
        log.warning("Optuna not available, skipping Optuna tuning")
        return results
    
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
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
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        return float(scores.mean())
    
    log.info("Running Optuna optimization (20 trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_improve")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    
    if len(study.trials) > 0:
        best_params = study.best_params
        best_value = study.best_value
        
        log.info(f"Best CV accuracy: {best_value:.4f}")
        log.info(f"Best params: {best_params}")
        
        # Test on holdout set
        result = train_and_evaluate(X_train, X_test, y_train, y_test, "Optuna_Best", best_params)
        acc = result["metrics"]["accuracy"]
        acc_diff = acc - baseline_acc
        
        log.info(f"Holdout Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
        
        results["optuna_best"] = {
            "params": best_params,
            "cv_accuracy": float(best_value),
            "metrics": result["metrics"],
            "accuracy_diff": float(acc_diff),
        }
    
    return results


def test_ensemble(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict:
    """Test ensemble methods."""
    log.info("\n" + "="*80)
    log.info("STRATEGY 4: Ensemble Methods")
    log.info("="*80)
    
    results = {}
    
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        **xgb_params,
    )
    
    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    # Voting ensemble
    log.info("\nTesting Voting Ensemble (XGBoost + RandomForest)...")
    voting_clf = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("rf", rf_clf)],
        voting="soft",
    )
    
    voting_clf.fit(X_train_scaled, y_train)
    y_pred = voting_clf.predict(X_test_scaled)
    y_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    acc_diff = acc - baseline_acc
    
    log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
    
    results["voting_ensemble"] = {
        "metrics": {
            "accuracy": acc,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "accuracy_diff": float(acc_diff),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Improve model validation accuracy")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/accuracy_improvement_results.json", help="Output JSON file")
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
    log.info("ACCURACY IMPROVEMENT EXPERIMENTS")
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
    
    # Baseline
    log.info("\n" + "="*80)
    log.info("BASELINE MODEL")
    log.info("="*80)
    baseline_start = time.perf_counter()
    baseline_result = train_and_evaluate(X_train, X_test, y_train, y_test, "Baseline")
    baseline_time = time.perf_counter() - baseline_start
    baseline_acc = baseline_result["metrics"]["accuracy"]
    
    log.info(f"\nBaseline Results (trained in {baseline_time:.2f}s):")
    log.info(f"  Accuracy: {baseline_acc:.4f}")
    log.info(f"  F1: {baseline_result['metrics']['f1']:.4f}")
    log.info(f"  ROC-AUC: {baseline_result['metrics']['roc_auc']:.4f}")
    
    # Test improvement strategies
    all_results = {
        "baseline": baseline_result["metrics"],
        "baseline_time": float(baseline_time),
        "strategies": {},
    }
    
    # Strategy 1: Feature Selection
    try:
        selection_results = test_feature_selection(X_train, X_test, y_train, y_test, baseline_acc)
        all_results["strategies"]["feature_selection"] = selection_results
    except Exception as e:
        log.error(f"Feature selection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Strategy 2: Feature Interactions
    try:
        interaction_results = test_feature_interactions(X_train, X_test, y_train, y_test, baseline_acc)
        all_results["strategies"]["feature_interactions"] = interaction_results
    except Exception as e:
        log.error(f"Feature interactions failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Strategy 3: Feature Transformations
    try:
        transform_results = test_feature_transformations(X_train, X_test, y_train, y_test, baseline_acc)
        all_results["strategies"]["feature_transformations"] = transform_results
    except Exception as e:
        log.error(f"Feature transformations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Strategy 4: Calibration
    try:
        calibration_results = test_calibration(X_train, X_test, y_train, y_test, baseline_acc)
        all_results["strategies"]["calibration"] = calibration_results
    except Exception as e:
        log.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Strategy 5: Optuna Tuning (if available)
    try:
        optuna_results = test_optuna_tuning(X_train, X_test, y_train, y_test, baseline_acc)
        if optuna_results:
            all_results["strategies"]["optuna_tuning"] = optuna_results
    except Exception as e:
        log.debug(f"Optuna tuning skipped: {e}")
    
    # Strategy 6: Ensemble
    try:
        ensemble_results = test_ensemble(X_train, X_test, y_train, y_test, baseline_acc)
        all_results["strategies"]["ensemble"] = ensemble_results
    except Exception as e:
        log.error(f"Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Find best improvement
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    best_acc = baseline_acc
    best_method = "baseline"
    all_improvements = []
    
    for strategy_name, strategy_results in all_results["strategies"].items():
        for method_name, method_result in strategy_results.items():
            if "accuracy_diff" in method_result:
                acc_diff = method_result["accuracy_diff"]
                acc = baseline_acc + acc_diff
                all_improvements.append({
                    "strategy": strategy_name,
                    "method": method_name,
                    "accuracy": acc,
                    "accuracy_diff": acc_diff,
                })
                if acc > best_acc:
                    best_acc = acc
                    best_method = f"{strategy_name}_{method_name}"
    
    log.info(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    log.info(f"Best Accuracy: {best_acc:.4f} ({best_acc - baseline_acc:+.4f}, {(best_acc - baseline_acc)*100:+.2f}%)")
    log.info(f"Best Method: {best_method}")
    
    if all_improvements:
        log.info("\nTop Improvements:")
        sorted_improvements = sorted(all_improvements, key=lambda x: x["accuracy"], reverse=True)
        for i, imp in enumerate(sorted_improvements[:10], 1):
            log.info(f"  {i}. {imp['strategy']}/{imp['method']}: {imp['accuracy']:.4f} ({imp['accuracy_diff']*100:+.2f}%)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            **all_results,
            "best_accuracy": float(best_acc),
            "best_method": best_method,
            "improvements": all_improvements,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

