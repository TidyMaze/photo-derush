#!/usr/bin/env python3
"""Feature ablation study: systematically test feature groups.

Usage:
    poetry run python scripts/feature_ablation.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
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
log = logging.getLogger("feature_ablation")

# Feature group definitions (indices for FAST mode with 71 features)
# These are approximate based on the feature extraction order
FEATURE_GROUPS = {
    "all": list(range(FEATURE_COUNT)),
    "geometric": [0, 1, 2, 3],  # width, height, aspect, log(file_size)
    "image_stats": [4, 5, 6, 7, 8, 9, 10, 11],  # brightness, std_brightness, mean/std RGB
    "histograms": list(range(12, 12 + (16 if USE_FULL_FEATURES else 8) * 3)),  # RGB histograms
    "quality": list(range(12 + (16 if USE_FULL_FEATURES else 8) * 3, 12 + (16 if USE_FULL_FEATURES else 8) * 3 + 5)),  # sharpness, saturation, entropy, highlight_clip, shadow_clip, noise
    "temporal": list(range(12 + (16 if USE_FULL_FEATURES else 8) * 3 + 5, 12 + (16 if USE_FULL_FEATURES else 8) * 3 + 9)),  # hour, day_of_week, month, is_weekend
    "exif": list(range(12 + (16 if USE_FULL_FEATURES else 8) * 3 + 9, FEATURE_COUNT)),  # ISO, aperture, shutter, flash, focal_length, etc.
}

# Adjust for actual feature count
if FEATURE_COUNT == 71:
    FEATURE_GROUPS["histograms"] = list(range(12, 36))  # 8 bins * 3 channels = 24 features
    FEATURE_GROUPS["quality"] = list(range(36, 42))  # sharpness, saturation, entropy, highlight_clip, shadow_clip, noise
    FEATURE_GROUPS["temporal"] = list(range(42, 46))  # hour, day_of_week, month, is_weekend
    FEATURE_GROUPS["exif"] = list(range(46, 71))  # remaining EXIF features
elif FEATURE_COUNT == 95:
    FEATURE_GROUPS["histograms"] = list(range(12, 60))  # 16 bins * 3 channels = 48 features
    FEATURE_GROUPS["quality"] = list(range(60, 66))
    FEATURE_GROUPS["temporal"] = list(range(66, 70))
    FEATURE_GROUPS["exif"] = list(range(70, 95))


def create_feature_mask(feature_groups_to_remove: list[str]) -> np.ndarray:
    """Create boolean mask for features to keep (True = keep, False = remove)."""
    mask = np.ones(FEATURE_COUNT, dtype=bool)

    for group_name in feature_groups_to_remove:
        if group_name in FEATURE_GROUPS:
            indices = FEATURE_GROUPS[group_name]
            mask[indices] = False
        else:
            log.warning(f"Unknown feature group: {group_name}")

    return mask


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_filenames: list[str],
    feature_mask: np.ndarray,
    group_name: str,
) -> dict:
    """Train model with masked features and return metrics."""
    # Apply feature mask
    X_train_masked = X_train[:, feature_mask]
    X_test_masked = X_test[:, feature_mask]

    n_features_used = int(feature_mask.sum())
    log.info(f"  Using {n_features_used}/{FEATURE_COUNT} features")

    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    # Compute class weights
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    # Build pipeline
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

    # Train
    clf.fit(X_train_masked, y_train)

    # Get feature importances
    feature_importances = None
    try:
        xgb_model = clf.named_steps.get("xgb")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            # Map back to original feature indices
            original_indices = np.where(feature_mask)[0]
            top_indices = np.argsort(importances)[::-1][:10]
            feature_importances = [(int(original_indices[i]), float(importances[i])) for i in top_indices]
    except Exception as e:
        log.debug(f"Could not extract feature importances: {e}")

    # Benchmark
    result = benchmark_model(
        clf=clf,
        X_train=X_train_masked,
        X_test=X_test_masked,
        y_train=y_train,
        y_test=y_test,
        test_filenames=test_filenames,
        model_name=f"XGBoost-{group_name}",
        model_path=None,
        hyperparameters=xgb_params,
        feature_importances=feature_importances,
        feature_count=n_features_used,
        feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
    )

    return {
        "group_name": group_name,
        "features_removed": FEATURE_COUNT - n_features_used,
        "features_used": n_features_used,
        "metrics": result.metrics,
        "feature_importances": feature_importances,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--out", default=None, help="Output JSON file (default: .cache/feature_ablation_results.json)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for train/test split")
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
        log.error(f"Insufficient labeled data: {len(y)} samples (need at least 20)")
        return 1

    # Fixed stratified train/test split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=args.random_state
    )

    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    log.info(f"Feature count: {FEATURE_COUNT} ({'FULL' if USE_FULL_FEATURES else 'FAST'} mode)")

    # Run ablation: test removing each feature group
    results = []

    # Baseline: all features
    log.info("\n=== Baseline: All Features ===")
    baseline_mask = create_feature_mask([])
    baseline_result = train_and_evaluate(
        X_train, X_test, y_train, y_test, filenames_test, baseline_mask, "all"
    )
    results.append(baseline_result)
    baseline_accuracy = baseline_result["metrics"]["accuracy"]

    # Remove each feature group
    groups_to_test = ["geometric", "image_stats", "histograms", "quality", "temporal", "exif"]
    for group_name in groups_to_test:
        log.info(f"\n=== Removing: {group_name} ===")
        mask = create_feature_mask([group_name])
        result = train_and_evaluate(X_train, X_test, y_train, y_test, filenames_test, mask, f"no_{group_name}")
        results.append(result)

        # Compare with baseline
        accuracy_delta = result["metrics"]["accuracy"] - baseline_accuracy
        log.info(f"  Accuracy delta: {accuracy_delta:+.4f}")

    # Print summary
    log.info("\n=== Summary ===")
    log.info(f"{'Group':<20} {'Features':<12} {'Accuracy':<12} {'Delta':<12} {'F1':<12}")
    log.info("-" * 70)

    for result in results:
        group_name = result["group_name"]
        features_used = result["features_used"]
        accuracy = result["metrics"]["accuracy"]
        f1 = result["metrics"]["f1"]
        delta = accuracy - baseline_accuracy if group_name != "all" else 0.0
        delta_str = f"{delta:+.4f}" if group_name != "all" else "baseline"
        log.info(f"{group_name:<20} {features_used:<12} {accuracy:<12.4f} {delta_str:<12} {f1:<12.4f}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "feature_ablation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

