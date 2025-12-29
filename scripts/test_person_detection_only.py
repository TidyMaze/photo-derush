#!/usr/bin/env python3
"""Test if person detection alone is useful as a feature.

This script tests:
1. Baseline (no object features)
2. Only person detection feature
3. All object detection features (for comparison)
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.object_detection import load_object_cache
from src.training_core import load_best_params
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)


def extract_person_feature(image_paths: list[str]) -> np.ndarray:
    """Extract only person detection feature."""
    log.info(f"Extracting person detection feature for {len(image_paths)} images...")
    
    cache = load_object_cache()
    person_classes = {"person"}
    
    features = []
    for path in image_paths:
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            features.append([0.0])
            continue
        
        classes = [d.get("class", "").lower() for d in detections]
        has_person = 1.0 if any(c in person_classes for c in classes) else 0.0
        features.append([has_person])
    
    return np.array(features, dtype=np.float32)


def extract_all_object_features(image_paths: list[str]) -> np.ndarray:
    """Extract all object detection features."""
    log.info(f"Extracting all object detection features for {len(image_paths)} images...")
    
    cache = load_object_cache()
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


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train and evaluate a model."""
    log.info(f"\n{'='*80}")
    log.info(f"Training {model_name}")
    log.info(f"{'='*80}")
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"Training samples: {len(y_train)} (keep={n_keep}, trash={n_trash})")
    log.info(f"Test samples: {len(y_test)}")
    log.info(f"Features: {X_train.shape[1]}")
    
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
    
    results = {
        "model_name": model_name,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
        "n_features": int(X_train.shape[1]),
    }
    
    log.info(f"\n{model_name} Results:")
    for metric, value in results["metrics"].items():
        log.info(f"  {metric:12s}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test person detection feature")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/Pictures/photo-dataset"), help="Image directory")
    parser.add_argument("--output", default=".cache/person_detection_test.json", help="Output JSON file")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("PERSON DETECTION FEATURE TEST")
    log.info("="*80)
    
    # Build dataset
    log.info(f"\nBuilding dataset from {args.image_dir}...")
    repo_path = os.path.join(args.image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return
    
    X_base, y, filenames = build_dataset(args.image_dir, repo=repo)
    
    if len(filenames) == 0:
        log.error("No labeled images found")
        return
    
    log.info(f"Found {len(filenames)} labeled images")
    log.info(f"  Keep: {np.sum(y == 1)}")
    log.info(f"  Trash: {np.sum(y == 0)}")
    
    X_base = np.array(X_base)
    y = np.array(y)
    paths = [os.path.join(args.image_dir, fname) for fname in filenames]
    
    # Extract features
    log.info("\nExtracting object detection features...")
    X_person = extract_person_feature(paths)
    X_all_obj = extract_all_object_features(paths)
    
    # Check person detection statistics
    person_count = np.sum(X_person == 1.0)
    log.info(f"\nPerson detection statistics:")
    log.info(f"  Images with person: {person_count}/{len(paths)} ({100*person_count/len(paths):.1f}%)")
    
    # Split data
    indices = np.arange(len(X_base))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_base = X_base[train_indices]
    X_test_base = X_base[test_indices]
    X_train_person = X_person[train_indices]
    X_test_person = X_person[test_indices]
    X_train_all_obj = X_all_obj[train_indices]
    X_test_all_obj = X_all_obj[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Train models
    results_baseline = train_and_evaluate(
        X_train_base, X_test_base, y_train, y_test, "Baseline (no object features)"
    )
    
    results_person = train_and_evaluate(
        np.hstack([X_train_base, X_train_person]), 
        np.hstack([X_test_base, X_test_person]),
        y_train, y_test, "With person detection only"
    )
    
    results_all_obj = train_and_evaluate(
        np.hstack([X_train_base, X_train_all_obj]),
        np.hstack([X_test_base, X_test_all_obj]),
        y_train, y_test, "With all object features"
    )
    
    # Compare
    log.info("\n" + "="*80)
    log.info("COMPARISON")
    log.info("="*80)
    
    log.info("\nMetric Comparison:")
    log.info(f"{'Metric':<15} {'Baseline':<15} {'Person Only':<15} {'All Objects':<15}")
    log.info("-" * 60)
    
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        baseline_val = results_baseline["metrics"][metric]
        person_val = results_person["metrics"][metric]
        all_obj_val = results_all_obj["metrics"][metric]
        log.info(f"{metric:<15} {baseline_val:<15.4f} {person_val:<15.4f} {all_obj_val:<15.4f}")
    
    # Person-only improvement
    person_acc_diff = results_person["metrics"]["accuracy"] - results_baseline["metrics"]["accuracy"]
    person_f1_diff = results_person["metrics"]["f1"] - results_baseline["metrics"]["f1"]
    
    log.info("\n" + "="*80)
    log.info("PERSON DETECTION ANALYSIS")
    log.info("="*80)
    log.info(f"\nPerson detection feature impact:")
    log.info(f"  Accuracy change: {person_acc_diff:+.4f} ({person_acc_diff*100:+.2f}%)")
    log.info(f"  F1 change: {person_f1_diff:+.4f} ({person_f1_diff*100:+.2f}%)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline": results_baseline,
            "person_only": results_person,
            "all_objects": results_all_obj,
            "person_statistics": {
                "images_with_person": int(person_count),
                "total_images": len(paths),
                "person_percentage": float(100*person_count/len(paths)),
            },
            "person_impact": {
                "accuracy_diff": float(person_acc_diff),
                "f1_diff": float(person_f1_diff),
            }
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    # Recommendation
    log.info("\n" + "="*80)
    log.info("RECOMMENDATION")
    log.info("="*80)
    
    if person_acc_diff > 0.01 and person_f1_diff > 0.01:
        log.info("✅ KEEP person detection feature")
        log.info(f"   Significant improvement: Accuracy +{person_acc_diff*100:.2f}%, F1 +{person_f1_diff*100:.2f}%")
    elif person_acc_diff > 0 or person_f1_diff > 0:
        log.info("⚠️  MARGINAL benefit from person detection")
        log.info(f"   Small improvement: Accuracy {person_acc_diff*100:+.2f}%, F1 {person_f1_diff*100:+.2f}%")
        log.info("   Consider keeping if feature extraction is fast and adds value")
    else:
        log.info("❌ REMOVE person detection feature")
        log.info(f"   Performance decrease: Accuracy {person_acc_diff*100:.2f}%, F1 {person_f1_diff*100:.2f}%")
        log.info("   Person detection does not improve model performance")


if __name__ == "__main__":
    main()

