#!/usr/bin/env python3
"""Study class balancing strategies: SMOTE, ADASYN, undersampling, cost-sensitive.

Usage:
    poetry run python scripts/class_balance_study.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("class_balance")


def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE (Synthetic Minority Oversampling Technique)."""
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        log.warning("imbalanced-learn not available, skipping SMOTE")
        return X, y


def apply_adasyn(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply ADASYN (Adaptive Synthetic Sampling)."""
    try:
        from imblearn.over_sampling import ADASYN

        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        log.warning("imbalanced-learn not available, skipping ADASYN")
        return X, y


def apply_undersampling(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply random undersampling of majority class."""
    try:
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        log.warning("imbalanced-learn not available, skipping undersampling")
        return X, y


def create_cost_sensitive_model(X_train: np.ndarray, y_train: np.ndarray, class_weight: dict) -> Pipeline:
    """Create cost-sensitive model with custom class weights."""
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight=class_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    **xgb_params,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Class balance study")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")

    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1

    log.info(f"Loading dataset from {image_dir}")

    # Initialize repository
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)

    # Build dataset
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 20:
        log.error(f"Insufficient labeled data: {len(y)} samples")
        return 1

    # Fixed stratified train/test split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=args.random_state
    )

    n_train_keep = int(np.sum(y_train == 1))
    n_train_trash = int(np.sum(y_train == 0))
    log.info(f"Train: {len(y_train)} samples (keep={n_train_keep}, trash={n_train_trash})")
    log.info(f"Test: {len(y_test)} samples")

    results = []

    # Baseline: standard XGBoost with scale_pos_weight
    log.info("\n=== Baseline: Standard XGBoost ===")
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    scale_pos_weight = n_train_trash / n_train_keep if n_train_keep > 0 else 1.0

    baseline_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    **xgb_params,
                ),
            ),
        ]
    )
    baseline_clf.fit(X_train, y_train)
    baseline_result = benchmark_model(
        baseline_clf,
        X_train,
        X_test,
        y_train,
        y_test,
        filenames_test,
        "Baseline",
        None,
        {},
        None,
        FEATURE_COUNT,
        "FULL" if USE_FULL_FEATURES else "FAST",
    )
    results.append({"method": "baseline", "metrics": baseline_result.metrics})
    log.info(f"Accuracy: {baseline_result.metrics['accuracy']:.4f}")
    log.info(f"Precision: {baseline_result.metrics['precision']:.4f}")
    log.info(f"Recall: {baseline_result.metrics['recall']:.4f}")

    # SMOTE
    log.info("\n=== SMOTE (Synthetic Minority Oversampling) ===")
    try:
        X_smote, y_smote = apply_smote(X_train, y_train)
        log.info(f"After SMOTE: {len(y_smote)} samples (keep={int(np.sum(y_smote==1))}, trash={int(np.sum(y_smote==0))})")

        smote_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **xgb_params,
                    ),
                ),
            ]
        )
        smote_clf.fit(X_smote, y_smote)
        smote_result = benchmark_model(
            smote_clf,
            X_smote,
            X_test,
            y_smote,
            y_test,
            filenames_test,
            "SMOTE",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "smote", "metrics": smote_result.metrics})
        log.info(f"Accuracy: {smote_result.metrics['accuracy']:.4f}")
        log.info(f"Precision: {smote_result.metrics['precision']:.4f}")
        log.info(f"Recall: {smote_result.metrics['recall']:.4f}")
    except Exception as e:
        log.error(f"SMOTE failed: {e}")

    # ADASYN
    log.info("\n=== ADASYN (Adaptive Synthetic Sampling) ===")
    try:
        X_adasyn, y_adasyn = apply_adasyn(X_train, y_train)
        log.info(f"After ADASYN: {len(y_adasyn)} samples")

        adasyn_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **xgb_params,
                    ),
                ),
            ]
        )
        adasyn_clf.fit(X_adasyn, y_adasyn)
        adasyn_result = benchmark_model(
            adasyn_clf,
            X_adasyn,
            X_test,
            y_adasyn,
            y_test,
            filenames_test,
            "ADASYN",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "adasyn", "metrics": adasyn_result.metrics})
        log.info(f"Accuracy: {adasyn_result.metrics['accuracy']:.4f}")
        log.info(f"Precision: {adasyn_result.metrics['precision']:.4f}")
        log.info(f"Recall: {adasyn_result.metrics['recall']:.4f}")
    except Exception as e:
        log.error(f"ADASYN failed: {e}")

    # Undersampling
    log.info("\n=== Random Undersampling ===")
    try:
        X_under, y_under = apply_undersampling(X_train, y_train)
        log.info(f"After undersampling: {len(y_under)} samples")

        under_clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **xgb_params,
                    ),
                ),
            ]
        )
        under_clf.fit(X_under, y_under)
        under_result = benchmark_model(
            under_clf,
            X_under,
            X_test,
            y_under,
            y_test,
            filenames_test,
            "Undersampling",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "undersampling", "metrics": under_result.metrics})
        log.info(f"Accuracy: {under_result.metrics['accuracy']:.4f}")
        log.info(f"Precision: {under_result.metrics['precision']:.4f}")
        log.info(f"Recall: {under_result.metrics['recall']:.4f}")
    except Exception as e:
        log.error(f"Undersampling failed: {e}")

    # Cost-sensitive learning (different class weights)
    log.info("\n=== Cost-Sensitive Learning (Class Weights) ===")
    for weight_ratio in [0.5, 1.0, 2.0, 5.0]:
        class_weight = {0: 1.0, 1: weight_ratio}  # Higher weight for keep class
        log.info(f"  Testing class_weight ratio: {weight_ratio}")
        try:
            cost_clf = create_cost_sensitive_model(X_train, y_train, class_weight)
            cost_clf.fit(X_train, y_train)
            cost_result = benchmark_model(
                cost_clf,
                X_train,
                X_test,
                y_train,
                y_test,
                filenames_test,
                f"CostSensitive_{weight_ratio}",
                None,
                {},
                None,
                FEATURE_COUNT,
                "FULL" if USE_FULL_FEATURES else "FAST",
            )
            results.append({"method": f"cost_sensitive_{weight_ratio}", "metrics": cost_result.metrics, "class_weight": weight_ratio})
        except Exception as e:
            log.error(f"Cost-sensitive (ratio={weight_ratio}) failed: {e}")

    # Print comparison
    log.info("\n=== Class Balance Strategy Comparison ===")
    log.info(f"{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    log.info("-" * 75)

    for result in results:
        method = result["method"]
        metrics = result["metrics"]
        accuracy = metrics.get("accuracy", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        log.info(f"{method:<25} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "class_balance_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

