#!/usr/bin/env python3
"""Compare different ML algorithms on the same dataset.

Usage:
    poetry run python scripts/compare_algorithms.py [IMAGE_DIR]
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("compare_algorithms")

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    log.warning("XGBoost not available")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    log.warning("LightGBM not available")

try:
    import catboost as cb
except ImportError:
    cb = None
    log.warning("CatBoost not available")


def create_xgboost_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Create XGBoost model."""
    if xgb is None:
        raise ImportError("XGBoost not available")

    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    return Pipeline(
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


def create_lightgbm_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Create LightGBM model."""
    if lgb is None:
        raise ImportError("LightGBM not available")

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lgb",
                lgb.LGBMClassifier(
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary",
                    metric="binary_logloss",
                    verbose=-1,
                ),
            ),
        ]
    )


def create_catboost_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Create CatBoost model."""
    if cb is None:
        raise ImportError("CatBoost not available")

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "cat",
                cb.CatBoostClassifier(
                    random_state=42,
                    thread_count=-1,
                    scale_pos_weight=scale_pos_weight,
                    loss_function="Logloss",
                    verbose=False,
                ),
            ),
        ]
    )


def create_random_forest_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Create Random Forest model."""
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    class_weight = "balanced" if n_keep > 0 and n_trash > 0 else None

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def create_mlp_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """Create Multi-Layer Perceptron model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Compare ML algorithms")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for train/test split")
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

    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    log.info(f"Features: {FEATURE_COUNT} ({'FULL' if USE_FULL_FEATURES else 'FAST'} mode)")

    # Algorithms to test
    algorithms = []

    if xgb:
        algorithms.append(("XGBoost", create_xgboost_model))
    if lgb:
        algorithms.append(("LightGBM", create_lightgbm_model))
    if cb:
        algorithms.append(("CatBoost", create_catboost_model))
    algorithms.append(("RandomForest", create_random_forest_model))
    algorithms.append(("MLP", create_mlp_model))

    results = []

    for algo_name, create_model_fn in algorithms:
        log.info(f"\n=== {algo_name} ===")
        try:
            # Create and train model
            clf = create_model_fn(X_train, y_train)
            log.info("Training...")
            clf.fit(X_train, y_train)

            # Get feature importances if available
            feature_importances = None
            try:
                if hasattr(clf, "named_steps"):
                    model_step = clf.named_steps.get(list(clf.named_steps.keys())[-1])
                    if model_step and hasattr(model_step, "feature_importances_"):
                        importances = model_step.feature_importances_
                        top_indices = np.argsort(importances)[::-1][:10]
                        feature_importances = [(int(idx), float(importances[idx])) for idx in top_indices]
            except Exception:
                pass

            # Benchmark
            benchmark_result = benchmark_model(
                clf=clf,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                test_filenames=filenames_test,
                model_name=algo_name,
                model_path=None,
                hyperparameters={},
                feature_importances=feature_importances,
                feature_count=FEATURE_COUNT,
                feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
            )

            results.append(
                {
                    "algorithm": algo_name,
                    "metrics": benchmark_result.metrics,
                    "feature_importances": feature_importances,
                }
            )

            log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
            log.info(f"F1: {benchmark_result.metrics['f1']:.4f}")
            log.info(f"ROC-AUC: {benchmark_result.metrics.get('roc_auc', 'N/A')}")

        except Exception as e:
            log.error(f"Failed to train {algo_name}: {e}", exc_info=True)
            results.append({"algorithm": algo_name, "error": str(e)})

    # Print comparison
    log.info("\n=== Algorithm Comparison ===")
    log.info(f"{'Algorithm':<15} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12} {'Precision':<12}")
    log.info("-" * 65)

    for result in results:
        if "error" in result:
            log.info(f"{result['algorithm']:<15} ERROR")
            continue

        algo_name = result["algorithm"]
        metrics = result["metrics"]
        accuracy = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        roc_auc = metrics.get("roc_auc", float("nan"))
        precision = metrics.get("precision", 0.0)

        roc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
        log.info(f"{algo_name:<15} {accuracy:<12.4f} {f1:<12.4f} {roc_str:<12} {precision:<12.4f}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "algorithm_comparison_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

