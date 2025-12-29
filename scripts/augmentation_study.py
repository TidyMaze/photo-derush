#!/usr/bin/env python3
"""Study data augmentation strategies for image classification.

Usage:
    poetry run python scripts/augmentation_study.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES, batch_extract_features
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("augmentation")


def augment_horizontal_flip(image_path: str, output_path: str) -> str:
    """Apply horizontal flip augmentation."""
    img = Image.open(image_path)
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(output_path)
    return output_path


def augment_rotation(image_path: str, output_path: str, angle: float) -> str:
    """Apply rotation augmentation."""
    img = Image.open(image_path)
    rotated = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
    rotated.save(output_path)
    return output_path


def augment_brightness(image_path: str, output_path: str, factor: float) -> str:
    """Apply brightness adjustment."""
    from PIL import ImageEnhance

    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    enhanced = enhancer.enhance(factor)
    enhanced.save(output_path)
    return output_path


def augment_contrast(image_path: str, output_path: str, factor: float) -> str:
    """Apply contrast adjustment."""
    from PIL import ImageEnhance

    img = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(img)
    enhanced = enhancer.enhance(factor)
    enhanced.save(output_path)
    return output_path


def augment_color_jitter(image_path: str, output_path: str, factor: float) -> str:
    """Apply color jitter (saturation and color balance)."""
    from PIL import ImageEnhance

    img = Image.open(image_path)
    # Saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.0 + factor)
    # Color balance (simplified)
    img.save(output_path)
    return output_path


def apply_augmentation(
    image_dir: str, filenames: list[str], labels: list[int], augmentation_fn, augmentation_name: str
) -> tuple[list[str], list[int]]:
    """Apply augmentation to training images."""
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="aug_")
    augmented_filenames = []
    augmented_labels = []

    log.info(f"Applying {augmentation_name} augmentation...")
    for fname, label in zip(filenames, labels):
        original_path = os.path.join(image_dir, fname)
        if not os.path.exists(original_path):
            continue

        # Create augmented version
        aug_fname = f"aug_{augmentation_name}_{fname}"
        aug_path = os.path.join(temp_dir, aug_fname)

        try:
            augmentation_fn(original_path, aug_path)
            augmented_filenames.append(aug_path)
            augmented_labels.append(label)
        except Exception as e:
            log.debug(f"Augmentation failed for {fname}: {e}")

    log.info(f"Created {len(augmented_filenames)} augmented images")
    return augmented_filenames, augmented_labels


def main():
    parser = argparse.ArgumentParser(description="Data augmentation study")
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

    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Baseline: no augmentation
    log.info("\n=== Baseline: No Augmentation ===")
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

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

    results = [{"method": "baseline", "metrics": baseline_result.metrics}]
    log.info(f"Accuracy: {baseline_result.metrics['accuracy']:.4f}")

    # Augmentation strategies
    augmentation_strategies = [
        ("horizontal_flip", lambda p, o: augment_horizontal_flip(p, o)),
        ("rotation_5", lambda p, o: augment_rotation(p, o, 5.0)),
        ("rotation_-5", lambda p, o: augment_rotation(p, o, -5.0)),
        ("brightness_1.1", lambda p, o: augment_brightness(p, o, 1.1)),
        ("brightness_0.9", lambda p, o: augment_brightness(p, o, 0.9)),
        ("contrast_1.1", lambda p, o: augment_contrast(p, o, 1.1)),
        ("contrast_0.9", lambda p, o: augment_contrast(p, o, 0.9)),
        ("color_jitter_0.1", lambda p, o: augment_color_jitter(p, o, 0.1)),
    ]

    # Get original training image paths
    train_paths = [os.path.join(image_dir, fname) for fname in filenames_train]

    for aug_name, aug_fn in augmentation_strategies:
        log.info(f"\n=== {aug_name} ===")
        try:
            # Apply augmentation
            aug_paths, aug_labels = apply_augmentation(image_dir, filenames_train, y_train.tolist(), aug_fn, aug_name)

            if len(aug_paths) == 0:
                log.warning(f"No augmented images created for {aug_name}")
                continue

            # Extract features from augmented images
            aug_features = batch_extract_features(aug_paths)
            X_aug = np.array([f for f in aug_features if f is not None], dtype=float)
            y_aug = np.array([aug_labels[i] for i, f in enumerate(aug_features) if f is not None], dtype=int)

            if len(X_aug) == 0:
                log.warning(f"No valid features extracted for {aug_name}")
                continue

            # Combine original and augmented training data
            X_train_combined = np.vstack([X_train, X_aug])
            y_train_combined = np.hstack([y_train, y_aug])

            log.info(f"Combined training set: {len(y_train_combined)} samples")

            # Train model
            aug_clf = Pipeline(
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
            aug_clf.fit(X_train_combined, y_train_combined)

            # Benchmark
            aug_result = benchmark_model(
                aug_clf,
                X_train_combined,
                X_test,
                y_train_combined,
                y_test,
                filenames_test,
                aug_name,
                None,
                {},
                None,
                FEATURE_COUNT,
                "FULL" if USE_FULL_FEATURES else "FAST",
            )

            results.append({"method": aug_name, "metrics": aug_result.metrics})
            log.info(f"Accuracy: {aug_result.metrics['accuracy']:.4f}")

            # Cleanup temp files
            import shutil

            temp_dir = os.path.dirname(aug_paths[0]) if aug_paths else None
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        except Exception as e:
            log.error(f"Augmentation {aug_name} failed: {e}", exc_info=True)

    # Print comparison
    log.info("\n=== Augmentation Comparison ===")
    log.info(f"{'Method':<25} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    log.info("-" * 75)

    for result in results:
        method = result["method"]
        metrics = result["metrics"]
        accuracy = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        log.info(f"{method:<25} {accuracy:<12.4f} {f1:<12.4f} {precision:<12.4f} {recall:<12.4f}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "augmentation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

