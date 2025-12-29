#!/usr/bin/env python3
"""Test each object detection feature individually to find which improve accuracy.

This script tests:
1. Baseline (no object features)
2. Each individual object feature separately
3. Reports which features help vs hurt performance
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


def extract_object_feature(image_paths: list[str], feature_name: str) -> np.ndarray:
    """Extract a single object detection feature."""
    cache = load_object_cache()
    
    # Define object categories
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
            features.append([0.0])
            continue
        
        classes = [d.get("class", "").lower() for d in detections]
        confidences = [d.get("confidence", 0.0) for d in detections]
        
        if feature_name == "num_objects":
            value = float(len(detections))
        elif feature_name == "max_confidence":
            value = max(confidences) if confidences else 0.0
        elif feature_name == "has_person":
            value = 1.0 if any(c in person_classes for c in classes) else 0.0
        elif feature_name == "has_animal":
            value = 1.0 if any(c in animal_classes for c in classes) else 0.0
        elif feature_name == "has_vehicle":
            value = 1.0 if any(c in vehicle_classes for c in classes) else 0.0
        elif feature_name == "has_food":
            value = 1.0 if any(c in food_classes for c in classes) else 0.0
        elif feature_name == "has_electronics":
            value = 1.0 if any(c in electronics_classes for c in classes) else 0.0
        elif feature_name == "object_diversity":
            value = float(len(set(classes)))
        else:
            value = 0.0
        
        features.append([value])
    
    return np.array(features, dtype=np.float32)


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train and evaluate a model."""
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
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Test individual object detection features")
    parser.add_argument("--image-dir", default=os.path.expanduser("~/Pictures/photo-dataset"), help="Image directory")
    parser.add_argument("--output", default=".cache/individual_object_features.json", help="Output JSON file")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("INDIVIDUAL OBJECT DETECTION FEATURE ANALYSIS")
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
    
    # Split data (same split for all models)
    indices = np.arange(len(X_base))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_base = X_base[train_indices]
    X_test_base = X_base[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Train baseline
    log.info("\nTraining baseline model...")
    results_baseline = train_and_evaluate(X_train_base, X_test_base, y_train, y_test, "Baseline")
    baseline_acc = results_baseline["metrics"]["accuracy"]
    baseline_f1 = results_baseline["metrics"]["f1"]
    
    log.info(f"Baseline Accuracy: {baseline_acc:.4f}")
    log.info(f"Baseline F1: {baseline_f1:.4f}")
    
    # Test each feature individually
    feature_names = [
        "num_objects",
        "max_confidence",
        "has_person",
        "has_animal",
        "has_vehicle",
        "has_food",
        "has_electronics",
        "object_diversity",
    ]
    
    log.info("\n" + "="*80)
    log.info("TESTING INDIVIDUAL FEATURES")
    log.info("="*80)
    
    results = {}
    improvements = []
    
    for feature_name in feature_names:
        log.info(f"\nTesting feature: {feature_name}")
        
        # Extract feature
        X_feat = extract_object_feature(paths, feature_name)
        X_train_feat = X_feat[train_indices]
        X_test_feat = X_feat[test_indices]
        
        # Check feature statistics
        feat_mean = np.mean(X_train_feat)
        feat_std = np.std(X_train_feat)
        feat_min = np.min(X_train_feat)
        feat_max = np.max(X_train_feat)
        non_zero_count = np.sum(X_train_feat != 0)
        
        log.info(f"  Statistics: mean={feat_mean:.3f}, std={feat_std:.3f}, range=[{feat_min:.3f}, {feat_max:.3f}]")
        log.info(f"  Non-zero values: {non_zero_count}/{len(X_train_feat)} ({100*non_zero_count/len(X_train_feat):.1f}%)")
        
        # Train with feature
        X_train_combined = np.hstack([X_train_base, X_train_feat])
        X_test_combined = np.hstack([X_test_base, X_test_feat])
        
        result = train_and_evaluate(X_train_combined, X_test_combined, y_train, y_test, feature_name)
        
        acc = result["metrics"]["accuracy"]
        f1 = result["metrics"]["f1"]
        acc_diff = acc - baseline_acc
        f1_diff = f1 - baseline_f1
        
        log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
        log.info(f"  F1: {f1:.4f} ({f1_diff:+.4f}, {f1_diff*100:+.2f}%)")
        
        results[feature_name] = {
            "metrics": result["metrics"],
            "statistics": {
                "mean": float(feat_mean),
                "std": float(feat_std),
                "min": float(feat_min),
                "max": float(feat_max),
                "non_zero_count": int(non_zero_count),
                "non_zero_percentage": float(100*non_zero_count/len(X_train_feat)),
            },
            "improvement": {
                "accuracy_diff": float(acc_diff),
                "f1_diff": float(f1_diff),
                "accuracy_pct": float(acc_diff * 100),
                "f1_pct": float(f1_diff * 100),
            }
        }
        
        improvements.append({
            "feature": feature_name,
            "accuracy_diff": acc_diff,
            "f1_diff": f1_diff,
            "accuracy": acc,
            "f1": f1,
        })
    
    # Sort by improvement
    improvements.sort(key=lambda x: x["accuracy_diff"], reverse=True)
    
    # Report results
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    log.info("\nFeatures ranked by accuracy improvement:")
    log.info(f"{'Feature':<20} {'Accuracy':<12} {'Acc Diff':<12} {'F1 Diff':<12} {'Status':<10}")
    log.info("-" * 70)
    
    for imp in improvements:
        acc_diff = imp["accuracy_diff"]
        f1_diff = imp["f1_diff"]
        status = "✅ HELPS" if acc_diff > 0.005 else ("⚠️  NEUTRAL" if acc_diff > -0.005 else "❌ HURTS")
        log.info(f"{imp['feature']:<20} {imp['accuracy']:<12.4f} {acc_diff:+12.4f} {f1_diff:+12.4f} {status:<10}")
    
    # Identify helpful features
    helpful_features = [imp for imp in improvements if imp["accuracy_diff"] > 0.005]
    harmful_features = [imp for imp in improvements if imp["accuracy_diff"] < -0.005]
    
    log.info("\n" + "="*80)
    log.info("RECOMMENDATIONS")
    log.info("="*80)
    
    if helpful_features:
        log.info("\n✅ HELPFUL FEATURES (improve accuracy):")
        for imp in helpful_features:
            log.info(f"  - {imp['feature']:<20} Accuracy: {imp['accuracy_diff']*100:+.2f}%, F1: {imp['f1_diff']*100:+.2f}%")
    else:
        log.info("\n⚠️  No features significantly improve accuracy")
    
    if harmful_features:
        log.info("\n❌ HARMFUL FEATURES (decrease accuracy):")
        for imp in harmful_features:
            log.info(f"  - {imp['feature']:<20} Accuracy: {imp['accuracy_diff']*100:.2f}%, F1: {imp['f1_diff']*100:.2f}%")
    
    # Test combination of helpful features
    if helpful_features:
        log.info("\n" + "="*80)
        log.info("TESTING COMBINATION OF HELPFUL FEATURES")
        log.info("="*80)
        
        helpful_feat_names = [imp["feature"] for imp in helpful_features]
        log.info(f"Combining: {', '.join(helpful_feat_names)}")
        
        # Extract all helpful features
        X_helpful_list = []
        for feat_name in helpful_feat_names:
            X_feat = extract_object_feature(paths, feat_name)
            X_helpful_list.append(X_feat[train_indices])
            X_helpful_list.append(X_feat[test_indices])
        
        X_train_helpful = np.hstack([X_train_base] + [X_feat[train_indices] for feat_name in helpful_feat_names])
        X_test_helpful = np.hstack([X_test_base] + [X_feat[test_indices] for feat_name in helpful_feat_names])
        
        result_combined = train_and_evaluate(X_train_helpful, X_test_helpful, y_train, y_test, "Combined helpful")
        
        combined_acc = result_combined["metrics"]["accuracy"]
        combined_f1 = result_combined["metrics"]["f1"]
        combined_acc_diff = combined_acc - baseline_acc
        combined_f1_diff = combined_f1 - baseline_f1
        
        log.info(f"\nCombined helpful features:")
        log.info(f"  Accuracy: {combined_acc:.4f} ({combined_acc_diff:+.4f}, {combined_acc_diff*100:+.2f}%)")
        log.info(f"  F1: {combined_f1:.4f} ({combined_f1_diff:+.4f}, {combined_f1_diff*100:+.2f}%)")
        
        results["combined_helpful"] = {
            "features": helpful_feat_names,
            "metrics": result_combined["metrics"],
            "improvement": {
                "accuracy_diff": float(combined_acc_diff),
                "f1_diff": float(combined_f1_diff),
            }
        }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline": results_baseline,
            "individual_features": results,
            "ranked_improvements": improvements,
            "helpful_features": [imp["feature"] for imp in helpful_features],
            "harmful_features": [imp["feature"] for imp in harmful_features],
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

