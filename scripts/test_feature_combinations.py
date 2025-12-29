#!/usr/bin/env python3
"""Test different feature combinations to improve model accuracy.

Usage:
    poetry run python scripts/test_feature_combinations.py [IMAGE_DIR]
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

from src.benchmark import benchmark_model, load_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("test_features")


def test_configuration(X: np.ndarray, y: np.ndarray, feature_indices: list[int], config_name: str) -> dict:
    """Test a feature configuration."""
    X_subset = X[:, feature_indices]
    
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X_subset, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    
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
    
    benchmark_result = benchmark_model(
        clf, X_train, X_test, y_train, y_test, filenames_test.tolist(),
        config_name, None, xgb_params, None, len(feature_indices), "FAST"
    )
    
    return {
        "config_name": config_name,
        "n_features": len(feature_indices),
        "feature_indices": feature_indices,
        "metrics": benchmark_result.metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Test feature combinations")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
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
    
    # Load baseline or compute it
    baseline = load_baseline()
    if baseline:
        baseline_acc = baseline.metrics["accuracy"]
        log.info(f"Baseline accuracy (from saved): {baseline_acc:.4f}")
    else:
        # Compute baseline with all features
        log.info("No baseline found, computing baseline with all features...")
        baseline_result = test_configuration(X, y, list(range(FEATURE_COUNT)), "baseline_all_features")
        baseline_acc = baseline_result["metrics"]["accuracy"]
        log.info(f"Baseline accuracy (computed): {baseline_acc:.4f}")
    
    results = []
    
    # Load feature selection results
    fs_path = Path(__file__).resolve().parent.parent / ".cache" / "feature_selection_results.json"
    top_features = None
    if fs_path.exists():
        with open(fs_path) as f:
            fs_data = json.load(f)
            if "xgboost_importance" in fs_data:
                top_features = [idx for idx, _ in fs_data["xgboost_importance"][:25]]
                log.info(f"Loaded top 25 features from feature selection")
    
    # Test configurations
    configs = [
        ("all_features", list(range(FEATURE_COUNT))),
    ]
    
    if top_features:
        configs.extend([
            ("top_25_features", top_features),
            ("top_20_features", top_features[:20]),
            ("top_15_features", top_features[:15]),
        ])
    
    # Based on ablation study: EXIF + Quality are critical
    exif_start = 46 if FEATURE_COUNT == 71 else 70
    exif_end = min(71, FEATURE_COUNT)
    quality_start = 36
    quality_end = min(42, FEATURE_COUNT)
    
    exif_indices = list(range(exif_start, exif_end))
    quality_indices = list(range(quality_start, quality_end))
    geometric_indices = list(range(4))  # First 4 features
    hist_indices = list(range(12, 36)) if FEATURE_COUNT == 71 else list(range(12, 60))
    
    configs.extend([
        ("exif_quality_only", sorted(set(exif_indices + quality_indices))),
        ("exif_quality_geometric", sorted(set(exif_indices + quality_indices + geometric_indices))),
        ("exif_quality_hist", sorted(set(exif_indices + quality_indices + hist_indices[:12]))),
        ("critical_features", sorted(set(exif_indices + quality_indices + geometric_indices + hist_indices[:8]))),
    ])
    
    best_acc = baseline_acc
    best_config = None
    
    for config_name, feature_indices in configs:
        log.info(f"\nTesting: {config_name} ({len(feature_indices)} features)")
        try:
            result = test_configuration(X, y, feature_indices, config_name)
            results.append(result)
            acc = result["metrics"]["accuracy"]
            log.info(f"  Accuracy: {acc:.4f} ({acc - baseline_acc:+.4f})")
            
            if acc > best_acc:
                best_acc = acc
                best_config = result
                log.info(f"  ✅ NEW BEST!")
        except Exception as e:
            log.error(f"  Failed: {e}")
    
    # Summary
    log.info("\n" + "="*60)
    log.info("FEATURE COMBINATION TEST SUMMARY")
    log.info("="*60)
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Best accuracy: {best_acc:.4f}")
    if baseline_acc > 0:
        log.info(f"Improvement: {best_acc - baseline_acc:+.4f} ({100*(best_acc - baseline_acc)/baseline_acc:+.2f}%)")
    else:
        log.info(f"Improvement: {best_acc - baseline_acc:+.4f} (baseline was 0, cannot compute percentage)")
    
    if best_config:
        log.info(f"\nBest configuration: {best_config['config_name']}")
        log.info(f"  Features: {best_config['n_features']}")
        log.info(f"  Accuracy: {best_config['metrics']['accuracy']:.4f}")
        log.info(f"  F1: {best_config['metrics']['f1']:.4f}")
        log.info(f"  ROC-AUC: {best_config['metrics'].get('roc_auc', 'N/A')}")
    
    # Print all results sorted by accuracy
    log.info("\nAll configurations (sorted by accuracy):")
    sorted_results = sorted(results, key=lambda x: x["metrics"]["accuracy"], reverse=True)
    for r in sorted_results:
        acc = r["metrics"]["accuracy"]
        delta = acc - baseline_acc
        log.info(f"  {r['config_name']:<30} {acc:.4f} ({delta:+.4f}) [{r['n_features']} features]")
    
    # Save results
    output_path = Path(__file__).resolve().parent.parent / ".cache" / "feature_combination_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "baseline_accuracy": baseline_acc,
            "best_accuracy": best_acc,
            "improvement": best_acc - baseline_acc,
            "results": results,
            "best_config": best_config,
        }, f, indent=2, default=str)
    
    log.info(f"\nSaved results to {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

