#!/usr/bin/env python3
"""Benchmark current model and prepare for AVA dataset integration.

This script:
1. Evaluates current model on existing dataset
2. Identifies areas for improvement
3. Prepares for AVA dataset integration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.inference import load_model, predict_keep_probability

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("benchmark")


def evaluate_model_on_dataset(image_dir: str, model_path: str = None):
    """Evaluate model on the current dataset."""
    log.info("="*80)
    log.info("MODEL BENCHMARKING")
    log.info("="*80)
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if not repo:
        log.error("No repository found")
        return None
    
    log.info(f"Loading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples")
    log.info(f"  Keep: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    log.info(f"  Trash: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    
    # Load model
    if model_path is None:
        from src.training_core import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
    
    log.info(f"\nLoading model from {model_path}...")
    bundle = load_model(model_path)
    if not bundle:
        log.error("Failed to load model")
        return None
    
    metadata = bundle.meta.get("__metadata__", {})
    log.info(f"Model: {metadata.get('model_type', 'Unknown')}")
    log.info(f"  Features: {bundle.meta.get('feature_length', 'Unknown')}")
    log.info(f"  Test accuracy (training): {metadata.get('test_accuracy', 0)*100:.2f}%")
    
    # Get full image paths
    image_paths = [os.path.join(image_dir, f) for f in filenames]
    
    # Predict
    log.info("\nRunning predictions...")
    probs = predict_keep_probability(image_paths, model_path=model_path)
    
    # Use optimal threshold
    threshold = bundle.meta.get("optimal_threshold", 0.67)
    predictions = (np.array(probs) >= threshold).astype(int)
    
    # Filter out NaN predictions
    valid_mask = ~np.isnan(probs)
    y_valid = y[valid_mask]
    pred_valid = predictions[valid_mask]
    prob_valid = np.array(probs)[valid_mask]
    
    log.info(f"Valid predictions: {np.sum(valid_mask)}/{len(probs)}")
    
    if len(y_valid) == 0:
        log.error("No valid predictions")
        return None
    
    # Metrics
    accuracy = accuracy_score(y_valid, pred_valid)
    precision = precision_score(y_valid, pred_valid, zero_division=0)
    recall = recall_score(y_valid, pred_valid, zero_division=0)
    f1 = f1_score(y_valid, pred_valid, zero_division=0)
    roc_auc = roc_auc_score(y_valid, prob_valid) if len(np.unique(y_valid)) > 1 else float('nan')
    cm = confusion_matrix(y_valid, pred_valid)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    log.info("\n" + "="*80)
    log.info("BENCHMARK RESULTS")
    log.info("="*80)
    log.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1:        {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC:   N/A")
    log.info(f"\nConfusion Matrix:")
    log.info(f"  True Negatives:  {tn}")
    log.info(f"  False Positives: {fp}")
    log.info(f"  False Negatives: {fn}")
    log.info(f"  True Positives:  {tp}")
    
    log.info(f"\nClassification Report:")
    log.info(classification_report(y_valid, pred_valid, target_names=['trash', 'keep']))
    
    # Error analysis
    log.info("\n" + "="*80)
    log.info("ERROR ANALYSIS")
    log.info("="*80)
    
    false_positives = np.where((y_valid == 0) & (pred_valid == 1))[0]
    false_negatives = np.where((y_valid == 1) & (pred_valid == 0))[0]
    
    log.info(f"False Positives (trash labeled as keep): {len(false_positives)}")
    log.info(f"False Negatives (keep labeled as trash): {len(false_negatives)}")
    
    if len(false_positives) > 0:
        fp_probs = prob_valid[false_positives]
        log.info(f"  FP probability range: {fp_probs.min():.3f} - {fp_probs.max():.3f} (mean: {fp_probs.mean():.3f})")
    
    if len(false_negatives) > 0:
        fn_probs = prob_valid[false_negatives]
        log.info(f"  FN probability range: {fn_probs.min():.3f} - {fn_probs.max():.3f} (mean: {fn_probs.mean():.3f})")
    
    # Save results
    results = {
        "model_path": model_path,
        "dataset_size": len(y),
        "valid_predictions": int(np.sum(valid_mask)),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "error_analysis": {
            "false_positives": int(len(false_positives)),
            "false_negatives": int(len(false_negatives)),
        },
        "threshold": float(threshold),
    }
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    results_path = cache_dir / "model_benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return results


def analyze_improvement_opportunities(results: dict):
    """Analyze results and suggest improvements."""
    log.info("\n" + "="*80)
    log.info("IMPROVEMENT OPPORTUNITIES")
    log.info("="*80)
    
    accuracy = results["metrics"]["accuracy"]
    fp = results["error_analysis"]["false_positives"]
    fn = results["error_analysis"]["false_negatives"]
    
    log.info(f"\nCurrent accuracy: {accuracy*100:.2f}%")
    
    suggestions = []
    
    # Check for class imbalance issues
    if fp > fn * 2:
        suggestions.append("High false positive rate - consider increasing threshold or adding trash examples")
    elif fn > fp * 2:
        suggestions.append("High false negative rate - consider decreasing threshold or adding keep examples")
    
    # Check for overfitting
    training_acc = results.get("training_accuracy", 0)
    if training_acc > 0 and accuracy < training_acc - 0.1:
        suggestions.append(f"Possible overfitting: training={training_acc*100:.1f}% vs test={accuracy*100:.1f}%")
    
    # General suggestions
    if accuracy < 0.85:
        suggestions.append("Consider using AVA dataset to increase training data (250k images)")
        suggestions.append("Try data augmentation to increase diversity")
    
    if accuracy >= 0.85:
        suggestions.append("Good accuracy - focus on edge cases and active learning")
        suggestions.append("Use AVA dataset to prevent overfitting and improve generalization")
    
    log.info("\nSuggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        log.info(f"  {i}. {suggestion}")
    
    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Benchmark current model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--model", default=None, help="Model path")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Directory not found: {image_dir}")
        return 1
    
    results = evaluate_model_on_dataset(image_dir, args.model)
    if not results:
        return 1
    
    analyze_improvement_opportunities(results)
    
    log.info("\n" + "="*80)
    log.info("NEXT STEPS FOR AVA INTEGRATION")
    log.info("="*80)
    log.info("1. Download AVA dataset metadata:")
    log.info("   git clone https://github.com/mtobeiyf/ava_downloader.git")
    log.info("   cp ava_downloader/AVA.txt .cache/ava_dataset/AVA_metadata.txt")
    log.info("")
    log.info("2. Prepare AVA labels:")
    log.info("   poetry run python scripts/download_ava_dataset.py --max-images 10000")
    log.info("")
    log.info("3. Download AVA images (requires ava_downloader tool)")
    log.info("")
    log.info("4. Train on combined dataset (current + AVA)")
    log.info("")
    log.info("5. Re-benchmark to measure improvement")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
