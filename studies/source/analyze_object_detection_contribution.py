#!/usr/bin/env python3
"""Analyze contribution of object detection features to prediction score.

This script:
1. Loads the current model and extracts feature importances
2. Checks if object detection features exist in the model
3. If not, adds object detection features and measures their contribution
4. Reports feature importance rankings and accuracy impact
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT, batch_extract_features
from src.model import RatingsTagsRepository
from src.object_detection import get_objects_for_images, load_object_cache
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


def load_current_model_importances(model_path: str) -> dict:
    """Load current model and extract feature importances."""
    log.info(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        return {}
    
    try:
        data = joblib.load(model_path)
        meta = data.get("__metadata__", {})
        
        # Try to get feature importances from model
        model = data.get("model")
        feature_importances = data.get("feature_importances")
        
        if feature_importances:
            log.info(f"Found {len(feature_importances)} feature importances in model")
            return {
                "feature_importances": feature_importances,
                "feature_count": meta.get("feature_count", FEATURE_COUNT),
                "n_samples": meta.get("n_samples"),
            }
        
        # Try to extract from model if it's XGBoost
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            feature_importances = [(int(idx), float(importances[idx])) for idx in top_indices]
            return {
                "feature_importances": feature_importances,
                "feature_count": len(importances),
                "n_samples": meta.get("n_samples"),
            }
        
        log.warning("Could not extract feature importances from model")
        return {}
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return {}


def train_with_object_features(X_base: np.ndarray, X_obj: np.ndarray, y: np.ndarray) -> dict:
    """Train model with and without object detection features."""
    log.info("Training models with/without object detection features...")
    
    # Combine features
    X_with_obj = np.hstack([X_base, X_obj])
    
    # Split data
    X_train_base, X_test_base, X_train_obj, X_test_obj, y_train, y_test = train_test_split(
        X_base, X_obj, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_with_obj = np.hstack([X_train_base, X_train_obj])
    X_test_with_obj = np.hstack([X_test_base, X_test_obj])
    
    # Load hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Train baseline (without object features)
    log.info("Training baseline model (without object features)...")
    clf_baseline = Pipeline([
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
    clf_baseline.fit(X_train_base, y_train)
    
    # Train with object features
    log.info("Training model with object detection features...")
    clf_with_obj = Pipeline([
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
    clf_with_obj.fit(X_train_with_obj, y_train)
    
    # Evaluate both
    y_pred_baseline = clf_baseline.predict(X_test_base)
    y_pred_with_obj = clf_with_obj.predict(X_test_with_obj)
    y_proba_baseline = clf_baseline.predict_proba(X_test_base)[:, 1]
    y_proba_with_obj = clf_with_obj.predict_proba(X_test_with_obj)[:, 1]
    
    metrics_baseline = {
        "accuracy": accuracy_score(y_test, y_pred_baseline),
        "f1": f1_score(y_test, y_pred_baseline),
        "precision": precision_score(y_test, y_pred_baseline),
        "recall": recall_score(y_test, y_pred_baseline),
        "roc_auc": roc_auc_score(y_test, y_proba_baseline),
    }
    
    metrics_with_obj = {
        "accuracy": accuracy_score(y_test, y_pred_with_obj),
        "f1": f1_score(y_test, y_pred_with_obj),
        "precision": precision_score(y_test, y_pred_with_obj),
        "recall": recall_score(y_test, y_pred_with_obj),
        "roc_auc": roc_auc_score(y_test, y_proba_with_obj),
    }
    
    # Extract object feature importances
    xgb_model = clf_with_obj.named_steps.get("xgb")
    obj_feature_importances = {}
    if xgb_model and hasattr(xgb_model, "feature_importances_"):
        importances = xgb_model.feature_importances_
        # Object features are the last 8 features
        obj_indices = list(range(len(importances) - 8, len(importances)))
        obj_feature_names = [
            "num_objects", "max_confidence", "has_person", "has_animal",
            "has_vehicle", "has_food", "has_electronics", "object_diversity"
        ]
        for idx, name in zip(obj_indices, obj_feature_names):
            obj_feature_importances[name] = float(importances[idx])
    
    return {
        "baseline": metrics_baseline,
        "with_object_features": metrics_with_obj,
        "object_feature_importances": obj_feature_importances,
        "improvement": {
            "accuracy": metrics_with_obj["accuracy"] - metrics_baseline["accuracy"],
            "f1": metrics_with_obj["f1"] - metrics_baseline["f1"],
            "roc_auc": metrics_with_obj["roc_auc"] - metrics_baseline["roc_auc"],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze object detection feature contribution")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/Pictures/photo-dataset"), help="Image directory")
    parser.add_argument("--model-path", default=os.path.expanduser("~/.photo-derush-keep-trash-model.joblib"), help="Model path")
    parser.add_argument("--output", default=".cache/object_detection_analysis.json", help="Output JSON file")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("OBJECT DETECTION FEATURE CONTRIBUTION ANALYSIS")
    log.info("="*80)
    
    # Load current model importances
    current_importances = load_current_model_importances(args.model_path)
    if current_importances:
        log.info("\nCurrent Model Feature Importances (Top 20):")
        log.info("-" * 80)
        for idx, importance in current_importances.get("feature_importances", [])[:20]:
            log.info(f"  Feature {idx:3d}: {importance:.6f}")
    
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
    
    # Build full paths
    paths = [os.path.join(args.image_dir, fname) for fname in filenames]
    
    log.info(f"Valid features: {len(X_base)}/{len(filenames)}")
    
    # Extract object detection features
    log.info("\nExtracting object detection features...")
    X_obj = extract_object_detection_features(paths)
    
    # Train and compare
    log.info("\nTraining models to compare...")
    results = train_with_object_features(X_base, X_obj, y)
    
    # Report results
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    
    log.info("\nBaseline (without object features):")
    for metric, value in results["baseline"].items():
        log.info(f"  {metric:12s}: {value:.4f}")
    
    log.info("\nWith Object Detection Features:")
    for metric, value in results["with_object_features"].items():
        log.info(f"  {metric:12s}: {value:.4f}")
    
    log.info("\nImprovement:")
    for metric, value in results["improvement"].items():
        sign = "+" if value >= 0 else ""
        log.info(f"  {metric:12s}: {sign}{value:.4f} ({sign}{value*100:.2f}%)")
    
    log.info("\nObject Detection Feature Importances:")
    log.info("-" * 80)
    sorted_obj_features = sorted(results["object_feature_importances"].items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_obj_features:
        log.info(f"  {name:20s}: {importance:.6f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "current_model": current_importances,
            "comparison": results,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    # Summary
    log.info("\n" + "="*80)
    log.info("SUMMARY")
    log.info("="*80)
    acc_improvement = results["improvement"]["accuracy"]
    if acc_improvement > 0.01:
        log.info(f"✅ Object detection features improve accuracy by {acc_improvement*100:.2f}%")
        log.info("   Recommendation: Add object detection features to model")
    elif acc_improvement > 0:
        log.info(f"⚠️  Object detection features improve accuracy by {acc_improvement*100:.2f}%")
        log.info("   Small improvement - may not be worth the added complexity")
    else:
        log.info(f"❌ Object detection features decrease accuracy by {abs(acc_improvement)*100:.2f}%")
        log.info("   Recommendation: Do not add object detection features")


if __name__ == "__main__":
    main()

