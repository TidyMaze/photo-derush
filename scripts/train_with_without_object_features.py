#!/usr/bin/env python3
"""Train and evaluate models with and without object detection features.

This script:
1. Trains a baseline model (without object detection features)
2. Trains a model with object detection features
3. Evaluates both models comprehensively
4. Provides recommendation on whether to keep object detection features
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT, batch_extract_features
from src.model import RatingsTagsRepository
from src.object_detection import load_object_cache
from src.training_core import load_best_params
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)


def extract_object_detection_features(image_paths: list[str]) -> np.ndarray:
    """Extract object detection features for images.
    
    Returns:
        Array of shape (n_images, n_features) where features are:
        - num_objects: total number of detected objects
        - max_confidence: highest confidence score
        - has_person: 1 if person detected, 0 otherwise
        - has_animal: 1 if any animal detected, 0 otherwise
        - has_vehicle: 1 if any vehicle detected, 0 otherwise
        - has_food: 1 if food detected, 0 otherwise
        - has_electronics: 1 if electronics detected, 0 otherwise
        - object_diversity: number of unique object classes
    """
    log.info(f"Extracting object detection features for {len(image_paths)} images...")
    
    # Load cache
    cache = load_object_cache()
    
    # Common object categories
    person_classes = {"person"}
    animal_classes = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    vehicle_classes = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
    food_classes = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
    electronics_classes = {"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "tv", "monitor"}
    
    features = []
    for path in image_paths:
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            # No detections - return zeros
            features.append([0.0] * 8)
            continue
        
        num_objects = len(detections)
        confidences = [d.get("confidence", 0.0) for d in detections]
        max_confidence = max(confidences) if confidences else 0.0
        
        classes = [d.get("class", "").lower() for d in detections]
        has_person = 1.0 if any(c in person_classes for c in classes) else 0.0
        has_animal = 1.0 if any(c in animal_classes for c in classes) else 0.0
        has_vehicle = 1.0 if any(c in vehicle_classes for c in classes) else 0.0
        has_food = 1.0 if any(c in food_classes for c in classes) else 0.0
        has_electronics = 1.0 if any(c in electronics_classes for c in classes) else 0.0
        object_diversity = float(len(set(classes)))
        
        features.append([
            float(num_objects),
            float(max_confidence),
            has_person,
            has_animal,
            has_vehicle,
            has_food,
            has_electronics,
            object_diversity,
        ])
    
    return np.array(features, dtype=np.float32)


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train and evaluate a model."""
    log.info(f"\n{'='*80}")
    log.info(f"Training {model_name}")
    log.info(f"{'='*80}")
    
    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"Training samples: {len(y_train)} (keep={n_keep}, trash={n_trash})")
    log.info(f"Test samples: {len(y_test)}")
    log.info(f"Features: {X_train.shape[1]}")
    
    # Train model
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
    
    log.info("Fitting model...")
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Feature importances
    feature_importances = None
    try:
        xgb_model = clf.named_steps.get("xgb")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            feature_importances = [(int(idx), float(importances[idx])) for idx in top_indices]
    except Exception as e:
        log.warning(f"Could not extract feature importances: {e}")
    
    results = {
        "model_name": model_name,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "feature_importances": feature_importances,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    
    log.info(f"\n{model_name} Results:")
    log.info(f"  Accuracy:  {accuracy:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall:   {recall:.4f}")
    log.info(f"  F1:       {f1:.4f}")
    log.info(f"  ROC-AUC:  {roc_auc:.4f}")
    log.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    if feature_importances:
        log.info(f"\n  Top 10 Feature Importances:")
        for idx, importance in feature_importances[:10]:
            log.info(f"    Feature {idx:3d}: {importance:.6f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train models with/without object detection features")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/Pictures/photo-dataset"), help="Image directory")
    parser.add_argument("--output", default=".cache/object_feature_comparison.json", help="Output JSON file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("OBJECT DETECTION FEATURE EVALUATION")
    log.info("="*80)
    
    # Build dataset
    log.info(f"\nBuilding dataset from {args.image_dir}...")
    repo_path = os.path.join(args.image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found - need labeled images")
        return
    
    X_base, y, filenames = build_dataset(args.image_dir, repo=repo)
    
    if len(filenames) == 0:
        log.error("No labeled images found")
        return
    
    log.info(f"Found {len(filenames)} labeled images")
    log.info(f"  Keep: {np.sum(y == 1)}")
    log.info(f"  Trash: {np.sum(y == 0)}")
    
    # Convert to numpy arrays
    X_base = np.array(X_base)
    y = np.array(y)
    
    # Build full paths for object detection
    paths = [os.path.join(args.image_dir, fname) for fname in filenames]
    
    # Extract object detection features
    log.info("\nExtracting object detection features...")
    X_obj = extract_object_detection_features(paths)
    
    # Combine features
    X_with_obj = np.hstack([X_base, X_obj])
    
    # Split data (same split for both models for fair comparison)
    log.info(f"\nSplitting data (test_size={args.test_size})...")
    indices = np.arange(len(X_base))
    train_indices, test_indices = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=y
    )
    X_train_base = X_base[train_indices]
    X_test_base = X_base[test_indices]
    X_train_obj = X_obj[train_indices]
    X_test_obj = X_obj[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    X_train_with_obj = np.hstack([X_train_base, X_train_obj])
    X_test_with_obj = np.hstack([X_test_base, X_test_obj])
    
    # Train baseline model (without object features)
    results_baseline = train_and_evaluate_model(
        X_train_base, X_test_base, y_train, y_test, "Baseline (without object detection)"
    )
    
    # Train model with object features
    results_with_obj = train_and_evaluate_model(
        X_train_with_obj, X_test_with_obj, y_train, y_test, "With Object Detection Features"
    )
    
    # Compare results
    log.info("\n" + "="*80)
    log.info("COMPARISON")
    log.info("="*80)
    
    metrics_baseline = results_baseline["metrics"]
    metrics_with_obj = results_with_obj["metrics"]
    
    log.info("\nMetric Comparison:")
    log.info(f"{'Metric':<15} {'Baseline':<15} {'With Objects':<15} {'Difference':<15}")
    log.info("-" * 60)
    
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        baseline_val = metrics_baseline[metric]
        with_obj_val = metrics_with_obj[metric]
        diff = with_obj_val - baseline_val
        sign = "+" if diff >= 0 else ""
        log.info(f"{metric:<15} {baseline_val:<15.4f} {with_obj_val:<15.4f} {sign}{diff:<14.4f}")
    
    # Extract object feature importances
    obj_feature_importances = {}
    if results_with_obj["feature_importances"]:
        # Object features are the last 8 features
        obj_feature_names = [
            "num_objects", "max_confidence", "has_person", "has_animal",
            "has_vehicle", "has_food", "has_electronics", "object_diversity"
        ]
        n_base_features = X_train_base.shape[1]
        for idx, importance in results_with_obj["feature_importances"]:
            obj_idx = idx - n_base_features
            if 0 <= obj_idx < len(obj_feature_names):
                obj_feature_importances[obj_feature_names[obj_idx]] = importance
    
    log.info("\nObject Detection Feature Importances:")
    log.info("-" * 80)
    if obj_feature_importances:
        sorted_obj_features = sorted(obj_feature_importances.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_obj_features:
            log.info(f"  {name:20s}: {importance:.6f}")
    else:
        log.info("  (No object features in top 20)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    comparison = {
        "baseline": results_baseline,
        "with_object_features": results_with_obj,
        "comparison": {
            "accuracy_diff": metrics_with_obj["accuracy"] - metrics_baseline["accuracy"],
            "precision_diff": metrics_with_obj["precision"] - metrics_baseline["precision"],
            "recall_diff": metrics_with_obj["recall"] - metrics_baseline["recall"],
            "f1_diff": metrics_with_obj["f1"] - metrics_baseline["f1"],
            "roc_auc_diff": metrics_with_obj["roc_auc"] - metrics_baseline["roc_auc"],
        },
        "object_feature_importances": obj_feature_importances,
    }
    
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    # Recommendation
    log.info("\n" + "="*80)
    log.info("RECOMMENDATION")
    log.info("="*80)
    
    acc_diff = comparison["comparison"]["accuracy_diff"]
    f1_diff = comparison["comparison"]["f1_diff"]
    roc_auc_diff = comparison["comparison"]["roc_auc_diff"]
    
    if acc_diff > 0.01 and f1_diff > 0.01:
        log.info(f"✅ KEEP object detection features")
        log.info(f"   Accuracy improvement: {acc_diff*100:+.2f}%")
        log.info(f"   F1 improvement: {f1_diff*100:+.2f}%")
        log.info(f"   ROC-AUC improvement: {roc_auc_diff*100:+.2f}%")
    elif acc_diff > 0 or f1_diff > 0:
        log.info(f"⚠️  MARGINAL benefit from object detection features")
        log.info(f"   Accuracy change: {acc_diff*100:+.2f}%")
        log.info(f"   F1 change: {f1_diff*100:+.2f}%")
        log.info(f"   Recommendation: Consider keeping if feature extraction is fast")
    else:
        log.info(f"❌ REMOVE object detection features")
        log.info(f"   Accuracy decrease: {acc_diff*100:.2f}%")
        log.info(f"   F1 decrease: {f1_diff*100:.2f}%")
        log.info(f"   ROC-AUC decrease: {roc_auc_diff*100:.2f}%")
        log.info(f"   Recommendation: Do not include object detection features in model")


if __name__ == "__main__":
    main()

