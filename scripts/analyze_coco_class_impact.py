#!/usr/bin/env python3
"""Analyze impact of each COCO class as a training feature.

For each COCO class, computes:
- Impact on keep-loss rate (primary optimization goal: < 2%)
- Impact on accuracy
- Impact on PR-AUC
- Feature importance ranking

Optimization function: Minimize keep-loss rate while maintaining accuracy.
Keep-loss rate = FN(keep→trash) / total_keep (target: < 2%)
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.object_detection import COCO_CLASSES, COCO_CLASSES_LIST, load_object_cache, get_cache_path
from src.training_core import _build_pipeline, _compute_asymmetric_metrics, _find_threshold_for_max_keep_loss

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_coco_class_feature(path: str, class_name: str, cache: dict) -> float:
    """Extract binary feature for a specific COCO class.
    
    Returns 1.0 if class is detected, 0.0 otherwise.
    """
    basename = os.path.basename(path)
    detections = cache.get(basename, [])
    if not detections:
        return 0.0
    
    classes = [d.get("class", "").lower() for d in detections]
    has_class = 1.0 if class_name.lower() in classes else 0.0
    return has_class


def compute_class_impact(class_name: str, X_base: np.ndarray, y: np.ndarray, 
                         image_paths: list[str], cache: dict, random_state: int = 42) -> dict:
    """Compute impact of adding a COCO class as a binary feature.
    
    Returns dict with metrics comparing baseline vs. with-class model.
    """
    # Extract binary feature for this class
    class_feature = np.array([extract_coco_class_feature(path, class_name, cache) for path in image_paths])
    
    # Check if class is present in dataset
    class_presence_rate = float(np.mean(class_feature))
    if class_presence_rate < 0.01:  # Less than 1% presence
        return {
            "class": class_name,
            "presence_rate": class_presence_rate,
            "impact": "insufficient_data",
            "keep_loss_improvement": 0.0,
            "accuracy_improvement": 0.0,
            "pr_auc_improvement": 0.0,
        }
    
    # Add class feature to feature matrix
    X_with_class = np.column_stack([X_base, class_feature])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_class, y, test_size=0.2, stratify=y, random_state=random_state
    )
    # Extract base features (all but last column)
    X_train_base = X_train[:, :-1]  # type: ignore
    X_test_base = X_test[:, :-1]  # type: ignore
    
    # Split train for early stopping (before building pipelines)
    if len(y_train) >= 20:
        X_train_fit_base, X_val_base, y_train_fit_base, y_val_base = train_test_split(
            X_train_base, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )
        X_train_fit_class, X_val_class, y_train_fit_class, y_val_class = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )
    else:
        X_train_fit_base, X_val_base, y_train_fit_base, y_val_base = X_train_base, X_train_base, y_train, y_train
        X_train_fit_class, X_val_class, y_train_fit_class, y_val_class = X_train, X_train, y_train, y_train
    
    # Train baseline model (without class)
    pipeline_base = _build_pipeline(
        random_state=random_state,
        scale_pos_weight=float(np.sum(y_train_fit_base == 0) / np.sum(y_train_fit_base == 1)) if np.sum(y_train_fit_base == 1) > 0 else 1.0,
        xgb_params={},
        early_stopping_rounds=50 if len(y_train) >= 20 else None,  # Disable early stopping if too small
        use_catboost=True,
        n_handcrafted_features=X_base.shape[1],
        fast_mode=True
    )
    
    # Fit with eval_set if early stopping enabled
    if len(y_train) >= 20 and hasattr(pipeline_base.named_steps.get('cat'), 'fit'):
        try:
            pipeline_base.fit(X_train_fit_base, y_train_fit_base, cat__eval_set=(X_val_base, y_val_base))
        except Exception:
            # Fallback: fit without eval_set
            pipeline_base.fit(X_train_fit_base, y_train_fit_base)
    else:
        pipeline_base.fit(X_train_fit_base, y_train_fit_base)
    
    # Train model with class feature
    pipeline_with_class = _build_pipeline(
        random_state=random_state,
        scale_pos_weight=float(np.sum(y_train_fit_class == 0) / np.sum(y_train_fit_class == 1)) if np.sum(y_train_fit_class == 1) > 0 else 1.0,
        xgb_params={},
        early_stopping_rounds=50 if len(y_train) >= 20 else None,  # Disable early stopping if too small
        use_catboost=True,
        n_handcrafted_features=X_with_class.shape[1],
        fast_mode=True
    )
    
    # Fit with eval_set if early stopping enabled
    if len(y_train) >= 20 and hasattr(pipeline_with_class.named_steps.get('cat'), 'fit'):
        try:
            pipeline_with_class.fit(X_train_fit_class, y_train_fit_class, cat__eval_set=(X_val_class, y_val_class))
        except Exception:
            # Fallback: fit without eval_set
            pipeline_with_class.fit(X_train_fit_class, y_train_fit_class)
    else:
        pipeline_with_class.fit(X_train_fit_class, y_train_fit_class)
    
    # Get predictions
    y_proba_base = pipeline_base.predict_proba(X_test_base)[:, 1]
    y_proba_with_class = pipeline_with_class.predict_proba(X_test)[:, 1]
    
    # Ensure numpy arrays
    y_test_arr = np.asarray(y_test)
    y_proba_base_arr = np.asarray(y_proba_base)
    y_proba_with_class_arr = np.asarray(y_proba_with_class)
    
    # Find optimal thresholds (minimize keep-loss < 2%)
    threshold_base = _find_threshold_for_max_keep_loss(y_test_arr, y_proba_base_arr, max_keep_loss=0.02)
    threshold_with_class = _find_threshold_for_max_keep_loss(y_test_arr, y_proba_with_class_arr, max_keep_loss=0.02)
    
    # Compute predictions with optimal thresholds
    y_pred_base = (y_proba_base_arr >= threshold_base).astype(int)
    y_pred_with_class = (y_proba_with_class_arr >= threshold_with_class).astype(int)
    
    # Compute metrics
    metrics_base = _compute_asymmetric_metrics(y_test_arr, y_pred_base, y_proba_base_arr)
    metrics_with_class = _compute_asymmetric_metrics(y_test_arr, y_pred_with_class, y_proba_with_class_arr)
    
    # Compute improvements
    keep_loss_improvement = metrics_base["keep_loss_rate"] - metrics_with_class["keep_loss_rate"]  # Positive = improvement
    keep_rate_improvement = metrics_base.get("keep_rate", 0.0) - metrics_with_class.get("keep_rate", 0.0)  # Positive = reduction (good)
    accuracy_improvement = metrics_with_class["accuracy"] - metrics_base["accuracy"]  # Positive = improvement
    pr_auc_improvement = metrics_with_class["pr_auc"] - metrics_base["pr_auc"]  # Positive = improvement
    
    return {
        "class": class_name,
        "presence_rate": class_presence_rate,
        "impact": "measured",
        "keep_loss_improvement": keep_loss_improvement,
        "keep_rate_improvement": keep_rate_improvement,  # Primary secondary metric (negative = good)
        "accuracy_improvement": accuracy_improvement,
        "pr_auc_improvement": pr_auc_improvement,
        "baseline_keep_loss": metrics_base["keep_loss_rate"],
        "with_class_keep_loss": metrics_with_class["keep_loss_rate"],
        "baseline_keep_rate": metrics_base.get("keep_rate", 0.0),
        "with_class_keep_rate": metrics_with_class.get("keep_rate", 0.0),
        "baseline_accuracy": metrics_base["accuracy"],
        "with_class_accuracy": metrics_with_class["accuracy"],
        "baseline_pr_auc": metrics_base["pr_auc"],
        "with_class_pr_auc": metrics_with_class["pr_auc"],
    }


def main():
    """Analyze impact of each COCO class."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze COCO class impact on model performance")
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, default="coco_class_impact.json", help="Output JSON file")
    args = parser.parse_args()
    
    logging.info("Loading dataset...")
    repo = RatingsTagsRepository(path=os.path.join(args.image_dir, ".ratings_tags.json"))
    X, y, filenames = build_dataset(args.image_dir, repo)
    
    if len(X) < 10:
        logging.error("Insufficient data for analysis (need at least 10 samples, got %d)", len(X))
        return
    
    # Convert filenames to full paths
    image_paths = [os.path.join(args.image_dir, fname) for fname in filenames]
    
    logging.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logging.info(f"Labels: {np.sum(y == 1)} keep, {np.sum(y == 0)} trash")
    
    # Load object detection cache
    logging.info("Loading object detection cache...")
    cache_path = get_cache_path()
    if not os.path.exists(cache_path):
        logging.error(f"Object detection cache not found: {cache_path}")
        logging.error("Run object detection first to generate cache")
        return
    
    cache = load_object_cache()
    logging.info(f"Loaded cache with {len(cache)} images")
    
    # Get all COCO classes (excluding background)
    coco_classes = [name for name in COCO_CLASSES_LIST[1:] if name]  # Skip background
    
    logging.info(f"Analyzing {len(coco_classes)} COCO classes...")
    
    results = []
    for idx, class_name in enumerate(coco_classes):
        logging.info(f"[{idx+1}/{len(coco_classes)}] Analyzing '{class_name}'...")
        try:
            impact = compute_class_impact(class_name, X, y, image_paths, cache)
            results.append(impact)
            
            if impact["impact"] == "measured":
                logging.info(
                    f"  Presence: {impact['presence_rate']:.1%}, "
                    f"Keep-loss Δ: {impact['keep_loss_improvement']:+.4f}, "
                    f"Accuracy Δ: {impact['accuracy_improvement']:+.4f}, "
                    f"PR-AUC Δ: {impact['pr_auc_improvement']:+.4f}"
                )
            else:
                logging.info(f"  Presence: {impact['presence_rate']:.1%}, {impact['impact']}")
        except Exception as e:
            logging.error(f"  Error analyzing '{class_name}': {e}")
            results.append({
                "class": class_name,
                "impact": "error",
                "error": str(e)
            })
    
    # Sort by keep-loss improvement (primary optimization goal)
    measured_results = [r for r in results if r.get("impact") == "measured"]
    measured_results.sort(key=lambda x: x.get("keep_loss_improvement", 0), reverse=True)
    
    # Print summary
    print("\n" + "="*80)
    print("COCO CLASS IMPACT ANALYSIS")
    print("="*80)
    print(f"\nOptimization Function: Minimize keep-loss rate (target: < 2%)")
    print(f"Secondary goals: Minimize keep_rate, Maximize PR-AUC")
    print(f"\nTop 15 classes by keep-loss improvement:")
    print(f"{'Class':<20} {'Presence':<10} {'Keep-loss Δ':<15} {'Keep-rate Δ':<15} {'Accuracy Δ':<15} {'PR-AUC Δ':<15}")
    print("-" * 100)
    
    for r in measured_results[:15]:
        keep_rate_delta = r.get('keep_rate_improvement', 0.0)
        print(
            f"{r['class']:<20} "
            f"{r['presence_rate']:>8.1%}  "
            f"{r['keep_loss_improvement']:>+13.4f}  "
            f"{keep_rate_delta:>+13.4f}  "
            f"{r['accuracy_improvement']:>+13.4f}  "
            f"{r['pr_auc_improvement']:>+13.4f}"
        )
    
    # Also show top classes by keep_rate reduction (primary secondary metric)
    # Note: keep_rate_improvement is positive when keep_rate is reduced (good)
    print(f"\nTop 10 classes by keep-rate reduction (primary secondary metric, higher is better):")
    print(f"{'Class':<20} {'Presence':<10} {'Keep-rate Δ':<15} {'Keep-loss Δ':<15} {'Accuracy Δ':<15}")
    print("-" * 80)
    sorted_by_keep_rate = sorted(measured_results, key=lambda x: x.get('keep_rate_improvement', 0.0), reverse=True)
    for r in sorted_by_keep_rate[:10]:
        keep_rate_delta = r.get('keep_rate_improvement', 0.0)
        print(
            f"{r['class']:<20} "
            f"{r['presence_rate']:>8.1%}  "
            f"{keep_rate_delta:>+13.4f}  "
            f"{r['keep_loss_improvement']:>+13.4f}  "
            f"{r['accuracy_improvement']:>+13.4f}"
        )
    
    # Save results
    import json
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            "optimization_function": "Minimize keep-loss rate (target: < 2%)",
            "secondary_goals": ["Maximize accuracy", "Maximize PR-AUC"],
            "results": results,
            "top_classes_by_keep_loss_improvement": [
                {k: v for k, v in r.items() if k != "baseline_keep_loss" and k != "with_class_keep_loss" and 
                 k != "baseline_accuracy" and k != "with_class_accuracy" and 
                 k != "baseline_pr_auc" and k != "with_class_pr_auc"}
                for r in measured_results[:20]
            ]
        }, f, indent=2)
    
    logging.info(f"\nResults saved to {output_path}")
    logging.info(f"Total classes analyzed: {len(results)}")
    logging.info(f"Classes with sufficient data: {len(measured_results)}")


if __name__ == "__main__":
    main()

