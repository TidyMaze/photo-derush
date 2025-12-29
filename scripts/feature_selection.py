#!/usr/bin/env python3
"""Feature selection: identify most important features using multiple methods.

Usage:
    poetry run python scripts/feature_selection.py [IMAGE_DIR] [--method METHOD]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("feature_selection")


def mutual_information_selection(X: np.ndarray, y: np.ndarray, top_k: int = 30) -> list[tuple[int, float]]:
    """Select top features using mutual information."""
    log.info("Computing mutual information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    top_indices = np.argsort(mi_scores)[::-1][:top_k]
    results = [(int(idx), float(mi_scores[idx])) for idx in top_indices]
    return results


def rfe_selection(
    X: np.ndarray, y: np.ndarray, n_features: int = 30, n_estimators: int = 100
) -> list[tuple[int, float]]:
    """Select features using Recursive Feature Elimination."""
    log.info(f"Running RFE with n_features={n_features}...")

    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    # Merge params, avoiding duplicate n_estimators
    merged_params = {k: v for k, v in xgb_params.items() if k != "n_estimators"}
    merged_params["n_estimators"] = n_estimators
    
    estimator = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        **merged_params,
    )

    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X, y)

    # Get rankings (1 = selected, higher = eliminated earlier)
    rankings = selector.ranking_
    selected_indices = np.where(selector.support_)[0]
    results = [(int(idx), float(1.0 / rankings[idx])) for idx in selected_indices]
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
    return results


def permutation_importance_selection(
    X: np.ndarray, y: np.ndarray, top_k: int = 30, n_estimators: int = 100
) -> list[tuple[int, float]]:
    """Select features using permutation importance."""
    log.info("Computing permutation importance...")

    from sklearn.inspection import permutation_importance

    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    # Merge params, avoiding duplicate n_estimators
    merged_params = {k: v for k, v in xgb_params.items() if k != "n_estimators"}
    merged_params["n_estimators"] = n_estimators
    
    clf = Pipeline(
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
                    **merged_params,
                ),
            ),
        ]
    )

    clf.fit(X, y)
    perm_importance = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    top_indices = np.argsort(perm_importance.importances_mean)[::-1][:top_k]
    results = [(int(idx), float(perm_importance.importances_mean[idx])) for idx in top_indices]
    return results


def correlation_analysis(X: np.ndarray, y: np.ndarray, threshold: float = 0.95) -> dict:
    """Analyze feature correlations to identify redundant features."""
    log.info(f"Analyzing feature correlations (threshold={threshold})...")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated feature pairs
    redundant_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > threshold:
                redundant_pairs.append((i, j, float(corr_matrix[i, j])))

    # Compute correlation with target (using point-biserial correlation)
    target_correlations = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        if np.std(feature) > 0:
            corr = float(np.corrcoef(feature, y)[0, 1])
            target_correlations.append((i, corr))

    target_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "redundant_pairs": redundant_pairs,
        "target_correlations": target_correlations[:30],  # Top 30
    }


def xgboost_importance(X: np.ndarray, y: np.ndarray, top_k: int = 30) -> list[tuple[int, float]]:
    """Get feature importances from XGBoost."""
    log.info("Computing XGBoost feature importances...")

    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    clf = Pipeline(
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

    clf.fit(X, y)
    xgb_model = clf.named_steps.get("xgb")
    if xgb_model and hasattr(xgb_model, "feature_importances_"):
        importances = xgb_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]
        results = [(int(idx), float(importances[idx])) for idx in top_indices]
        return results
    return []


def main():
    parser = argparse.ArgumentParser(description="Feature selection analysis")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--method", choices=["all", "mi", "rfe", "perm", "corr", "xgb"], default="all", help="Selection method")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top features to select")
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

    log.info(f"Dataset: {len(y)} samples, {FEATURE_COUNT} features")

    results = {}

    if args.method in ["all", "mi"]:
        log.info("\n=== Mutual Information Selection ===")
        mi_results = mutual_information_selection(X, y, top_k=args.top_k)
        results["mutual_information"] = mi_results
        log.info(f"Top {min(10, len(mi_results))} features by MI:")
        for idx, score in mi_results[:10]:
            log.info(f"  Feature {idx}: {score:.4f}")

    if args.method in ["all", "rfe"]:
        log.info("\n=== Recursive Feature Elimination ===")
        rfe_results = rfe_selection(X, y, n_features=args.top_k)
        results["rfe"] = rfe_results
        log.info(f"Top {min(10, len(rfe_results))} features by RFE:")
        for idx, score in rfe_results[:10]:
            log.info(f"  Feature {idx}: {score:.4f}")

    if args.method in ["all", "perm"]:
        log.info("\n=== Permutation Importance ===")
        perm_results = permutation_importance_selection(X, y, top_k=args.top_k)
        results["permutation_importance"] = perm_results
        log.info(f"Top {min(10, len(perm_results))} features by Permutation Importance:")
        for idx, score in perm_results[:10]:
            log.info(f"  Feature {idx}: {score:.4f}")

    if args.method in ["all", "corr"]:
        log.info("\n=== Correlation Analysis ===")
        corr_results = correlation_analysis(X, y)
        results["correlation"] = corr_results
        log.info(f"Found {len(corr_results['redundant_pairs'])} redundant feature pairs")
        log.info(f"Top {min(10, len(corr_results['target_correlations']))} features by target correlation:")
        for idx, corr in corr_results["target_correlations"][:10]:
            log.info(f"  Feature {idx}: {corr:.4f}")

    if args.method in ["all", "xgb"]:
        log.info("\n=== XGBoost Feature Importance ===")
        xgb_results = xgboost_importance(X, y, top_k=args.top_k)
        results["xgboost_importance"] = xgb_results
        log.info(f"Top {min(10, len(xgb_results))} features by XGBoost importance:")
        for idx, score in xgb_results[:10]:
            log.info(f"  Feature {idx}: {score:.4f}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "feature_selection_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

