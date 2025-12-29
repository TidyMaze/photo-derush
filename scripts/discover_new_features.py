#!/usr/bin/env python3
"""Systematically discover new features that improve validation accuracy.

Tests various feature candidates:
- Additional object detection features (spatial, confidence, diversity)
- Image complexity metrics
- Texture features
- Color distribution features
- Composition features

Usage:
    poetry run python scripts/discover_new_features.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.features import extract_features
from src.model import RatingsTagsRepository
from src.object_detection import load_object_cache
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("discover_features")


def extract_object_spatial_features(image_paths: list[str]) -> np.ndarray:
    """Extract spatial features from object detections.
    
    Features:
    - object_center_x: normalized center X coordinate (0-1)
    - object_center_y: normalized center Y coordinate (0-1)
    - object_size_ratio: average object size relative to image
    - object_spread: spatial spread of objects
    """
    log.info(f"[object_spatial] Starting extraction for {len(image_paths)} images...")
    start_time = time.perf_counter()
    last_log_time = start_time
    cache = load_object_cache()
    log.info(f"[object_spatial] Loaded object cache: {len(cache)} entries")
    features = []
    
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        # Log every ~1 second or at milestones
        if current_time - last_log_time >= 1.0 or (idx + 1) % 100 == 0 or idx == 0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (idx + 1) / len(image_paths)
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            log.info(f"[object_spatial] {idx + 1}/{len(image_paths)} ({pct:.1f}%) - {rate:.1f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            features.append([0.5, 0.5, 0.0, 0.0])  # Defaults
            continue
        
        # Get image dimensions from first detection
        det_w = detections[0].get("det_w", 800)
        det_h = detections[0].get("det_h", 600)
        
        centers_x = []
        centers_y = []
        sizes = []
        
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2.0 / det_w if det_w > 0 else 0.5
                center_y = (y1 + y2) / 2.0 / det_h if det_h > 0 else 0.5
                size = ((x2 - x1) * (y2 - y1)) / (det_w * det_h) if det_w * det_h > 0 else 0.0
                centers_x.append(center_x)
                centers_y.append(center_y)
                sizes.append(size)
        
        if centers_x:
            center_x_mean = float(np.mean(centers_x))
            center_y_mean = float(np.mean(centers_y))
            size_ratio = float(np.mean(sizes))
            # Spatial spread: std of centers
            spread = float(np.std(centers_x) + np.std(centers_y)) / 2.0
        else:
            center_x_mean, center_y_mean, size_ratio, spread = 0.5, 0.5, 0.0, 0.0
        
        features.append([center_x_mean, center_y_mean, size_ratio, spread])
    
    elapsed = time.perf_counter() - start_time
    log.info(f"[object_spatial] Completed in {elapsed:.2f}s ({elapsed/len(image_paths)*1000:.1f}ms per image)")
    return np.array(features, dtype=np.float32)


def extract_object_confidence_features(image_paths: list[str]) -> np.ndarray:
    """Extract confidence-based features from object detections.
    
    Features:
    - mean_confidence: average confidence across all detections
    - confidence_std: std of confidence scores
    - high_confidence_count: number of detections with confidence > 0.7
    """
    log.info(f"[object_confidence] Starting extraction for {len(image_paths)} images...")
    start_time = time.perf_counter()
    last_log_time = start_time
    cache = load_object_cache()
    features = []
    
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        if current_time - last_log_time >= 1.0 or (idx + 1) % 100 == 0 or idx == 0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (idx + 1) / len(image_paths)
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            log.info(f"[object_confidence] {idx + 1}/{len(image_paths)} ({pct:.1f}%) - {rate:.1f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            features.append([0.0, 0.0, 0.0])
            continue
        
        confidences = [d.get("confidence", 0.0) for d in detections]
        mean_conf = float(np.mean(confidences))
        std_conf = float(np.std(confidences)) if len(confidences) > 1 else 0.0
        high_conf_count = float(sum(1 for c in confidences if c > 0.7))
        
        features.append([mean_conf, std_conf, high_conf_count])
    
    elapsed = time.perf_counter() - start_time
    log.info(f"[object_confidence] Completed in {elapsed:.2f}s ({elapsed/len(image_paths)*1000:.1f}ms per image)")
    return np.array(features, dtype=np.float32)


def extract_object_category_features(image_paths: list[str]) -> np.ndarray:
    """Extract category-specific features from object detections.
    
    Features:
    - has_animal: 1 if any animal detected
    - has_vehicle: 1 if any vehicle detected
    - has_food: 1 if any food detected
    - has_electronics: 1 if any electronics detected
    - has_furniture: 1 if any furniture detected
    - category_diversity: number of unique categories
    """
    log.info(f"[object_category] Starting extraction for {len(image_paths)} images...")
    start_time = time.perf_counter()
    last_log_time = start_time
    cache = load_object_cache()
    
    animal_classes = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    vehicle_classes = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}
    food_classes = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
    electronics_classes = {"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "tv", "monitor"}
    furniture_classes = {"chair", "couch", "bed", "dining table"}
    
    features = []
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        if current_time - last_log_time >= 1.0 or (idx + 1) % 100 == 0 or idx == 0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (idx + 1) / len(image_paths)
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            log.info(f"[object_category] {idx + 1}/{len(image_paths)} ({pct:.1f}%) - {rate:.1f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        
        classes = [d.get("class", "").lower() for d in detections]
        has_animal = 1.0 if any(c in animal_classes for c in classes) else 0.0
        has_vehicle = 1.0 if any(c in vehicle_classes for c in classes) else 0.0
        has_food = 1.0 if any(c in food_classes for c in classes) else 0.0
        has_electronics = 1.0 if any(c in electronics_classes for c in classes) else 0.0
        has_furniture = 1.0 if any(c in furniture_classes for c in classes) else 0.0
        category_diversity = float(len(set(classes)))
        
        features.append([has_animal, has_vehicle, has_food, has_electronics, has_furniture, category_diversity])
    
    elapsed = time.perf_counter() - start_time
    log.info(f"[object_category] Completed in {elapsed:.2f}s ({elapsed/len(image_paths)*1000:.1f}ms per image)")
    return np.array(features, dtype=np.float32)


def extract_image_complexity_features(image_paths: list[str]) -> np.ndarray:
    """Extract image complexity features.
    
    Features:
    - edge_density_ratio: ratio of edge pixels to total pixels
    - texture_variance: variance of local texture patterns
    - gradient_magnitude_mean: mean gradient magnitude
    """
    log.info(f"[image_complexity] Starting extraction for {len(image_paths)} images...")
    log.info(f"[image_complexity] Processing images sequentially (this may take a while)...")
    start_time = time.perf_counter()
    last_log_time = start_time
    features = []
    
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        # Log every ~1 second (more frequent for slow operations)
        if current_time - last_log_time >= 1.0 or (idx + 1) % 20 == 0 or idx == 0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (idx + 1) / len(image_paths)
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            log.info(f"[image_complexity] {idx + 1}/{len(image_paths)} ({pct:.1f}%) - {rate:.2f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            gray = np.mean(arr, axis=2)
            
            # Edge density: use Sobel-like gradient
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            magnitude = np.sqrt(grad_x[:, :-1] ** 2 + grad_y[:-1, :] ** 2)
            
            edge_threshold = np.percentile(magnitude, 75) if magnitude.size > 0 else 0.0
            edge_density = float(np.mean(magnitude > edge_threshold)) if magnitude.size > 0 else 0.0
            
            # Texture variance: local variance
            h, w = gray.shape
            if h > 4 and w > 4:
                # Sample local patches
                patch_size = min(8, h // 4, w // 4)
                patches = []
                for i in range(0, h - patch_size, patch_size):
                    for j in range(0, w - patch_size, patch_size):
                        patch = gray[i:i+patch_size, j:j+patch_size]
                        patches.append(np.var(patch))
                texture_variance = float(np.mean(patches)) if patches else 0.0
            else:
                texture_variance = float(np.var(gray))
            
            # Gradient magnitude mean
            grad_magnitude_mean = float(np.mean(magnitude)) if magnitude.size > 0 else 0.0
            
            features.append([edge_density, texture_variance, grad_magnitude_mean])
        except Exception as e:
            log.debug(f"Failed to extract complexity features for {path}: {e}")
            features.append([0.0, 0.0, 0.0])
    
    elapsed = time.perf_counter() - start_time
    log.info(f"[image_complexity] Completed in {elapsed:.2f}s ({elapsed/len(image_paths)*1000:.1f}ms per image)")
    return np.array(features, dtype=np.float32)


def extract_color_distribution_features(image_paths: list[str]) -> np.ndarray:
    """Extract color distribution features.
    
    Features:
    - dominant_color_ratio: ratio of dominant color pixels
    - color_uniformity: how uniform color distribution is
    - saturation_variance: variance of saturation values
    """
    log.info(f"[color_distribution] Starting extraction for {len(image_paths)} images...")
    log.info(f"[color_distribution] Processing images sequentially (this may take a while)...")
    start_time = time.perf_counter()
    last_log_time = start_time
    features = []
    
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        # Log every ~1 second (more frequent for slow operations)
        if current_time - last_log_time >= 1.0 or (idx + 1) % 20 == 0 or idx == 0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            pct = 100 * (idx + 1) / len(image_paths)
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            log.info(f"[color_distribution] {idx + 1}/{len(image_paths)} ({pct:.1f}%) - {rate:.2f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            h, w, _ = arr.shape
            
            # Reshape to pixels
            pixels = arr.reshape(-1, 3)
            
            # Dominant color: quantize and find most common
            quantized = (pixels / 32).astype(int)
            unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
            if len(counts) > 0:
                dominant_ratio = float(np.max(counts) / len(pixels))
            else:
                dominant_ratio = 0.0
            
            # Color uniformity: entropy of color distribution
            if len(counts) > 0:
                probs = counts / counts.sum()
                color_uniformity = float(-np.sum(probs * np.log2(probs + 1e-12)))
            else:
                color_uniformity = 0.0
            
            # Saturation variance
            max_vals = np.max(pixels, axis=1)
            min_vals = np.min(pixels, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                saturation = np.where(max_vals > 0, (max_vals - min_vals) / max_vals, 0.0)
            saturation_variance = float(np.var(saturation))
            
            features.append([dominant_ratio, color_uniformity, saturation_variance])
        except Exception as e:
            log.debug(f"Failed to extract color features for {path}: {e}")
            features.append([0.0, 0.0, 0.0])
    
    elapsed = time.perf_counter() - start_time
    log.info(f"[color_distribution] Completed in {elapsed:.2f}s ({elapsed/len(image_paths)*1000:.1f}ms per image)")
    return np.array(features, dtype=np.float32)


def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Train and evaluate a model."""
    log.info(f"[train_eval] Training model: {model_name}")
    log.info(f"[train_eval] Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    log.info(f"[train_eval] Test set: {len(X_test)} samples")
    
    train_start = time.perf_counter()
    last_log_time = train_start
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    log.info(f"[train_eval] Class balance: {n_keep} keep, {n_trash} trash (scale_pos_weight={scale_pos_weight:.2f})")
    
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
    
    log.info(f"[train_eval] Fitting model...")
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - train_start
    log.info(f"[train_eval] Training completed in {train_time:.2f}s")
    
    log.info(f"[train_eval] Making predictions...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Discover new features that improve accuracy")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/feature_discovery_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("FEATURE DISCOVERY: Testing New Feature Candidates")
    log.info("="*80)
    
    # Build baseline dataset
    log.info(f"\n{'='*80}")
    log.info("PHASE 1: Building baseline dataset")
    log.info(f"{'='*80}")
    dataset_start = time.perf_counter()
    log.info(f"Loading dataset from {image_dir}...")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info("Extracting features from images (this may take a while)...")
    log.info("Note: Feature extraction progress is logged by the build_dataset function")
    X_base, y, filenames = build_dataset(image_dir, repo=repo)
    dataset_time = time.perf_counter() - dataset_start
    
    if len(filenames) == 0:
        log.error("No labeled images found")
        return 1
    
    log.info(f"Dataset built in {dataset_time:.2f}s")
    log.info(f"Found {len(filenames)} labeled images")
    log.info(f"  Keep: {np.sum(y == 1)}")
    log.info(f"  Trash: {np.sum(y == 0)}")
    log.info(f"  Features per image: {X_base.shape[1] if len(X_base) > 0 else 0}")
    
    X_base = np.array(X_base)
    y = np.array(y)
    paths = [os.path.join(image_dir, fname) for fname in filenames]
    
    # Fixed split
    indices = np.arange(len(X_base))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_base = X_base[train_indices]
    X_test_base = X_base[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Train baseline
    log.info(f"\n{'='*80}")
    log.info("PHASE 2: Training baseline model")
    log.info(f"{'='*80}")
    baseline_start = time.perf_counter()
    results_baseline = train_and_evaluate(X_train_base, X_test_base, y_train, y_test, "Baseline")
    baseline_time = time.perf_counter() - baseline_start
    baseline_acc = results_baseline["metrics"]["accuracy"]
    baseline_f1 = results_baseline["metrics"]["f1"]
    
    log.info(f"\nBaseline Results (trained in {baseline_time:.2f}s):")
    log.info(f"  Accuracy: {baseline_acc:.4f}")
    log.info(f"  F1: {baseline_f1:.4f}")
    log.info(f"  Precision: {results_baseline['metrics']['precision']:.4f}")
    log.info(f"  Recall: {results_baseline['metrics']['recall']:.4f}")
    log.info(f"  ROC-AUC: {results_baseline['metrics']['roc_auc']:.4f}")
    
    # Feature groups to test
    feature_groups = {
        "object_spatial": (extract_object_spatial_features, 4),
        "object_confidence": (extract_object_confidence_features, 3),
        "object_category": (extract_object_category_features, 6),
        "image_complexity": (extract_image_complexity_features, 3),
        "color_distribution": (extract_color_distribution_features, 3),
    }
    
    log.info(f"\n{'='*80}")
    log.info("PHASE 3: Testing feature groups")
    log.info(f"{'='*80}")
    log.info(f"Will test {len(feature_groups)} feature groups:")
    for group_name, (_, n_features) in feature_groups.items():
        log.info(f"  - {group_name}: {n_features} features")
    
    results = {}
    improvements = []
    total_start = time.perf_counter()
    last_status_log = total_start
    
    for group_idx, (group_name, (extract_fn, n_features)) in enumerate(feature_groups.items(), 1):
        log.info(f"\n{'='*60}")
        log.info(f"Testing feature group {group_idx}/{len(feature_groups)}: {group_name} ({n_features} features)")
        log.info(f"{'='*60}")
        group_start = time.perf_counter()
        
        try:
            # Extract features
            log.info(f"[{group_name}] Extracting features...")
            extract_start = time.perf_counter()
            X_feat = extract_fn(paths)
            extract_time = time.perf_counter() - extract_start
            log.info(f"[{group_name}] Feature extraction completed in {extract_time:.2f}s")
            X_train_feat = X_feat[train_indices]
            X_test_feat = X_feat[test_indices]
            
            # Check feature statistics
            feat_mean = np.mean(X_train_feat, axis=0)
            feat_std = np.std(X_train_feat, axis=0)
            log.info(f"[{group_name}] Feature statistics:")
            log.info(f"  Mean: {feat_mean}")
            log.info(f"  Std:  {feat_std}")
            
            # Train with feature group
            log.info(f"[{group_name}] Training model with new features...")
            train_start_time = time.perf_counter()
            X_train_combined = np.hstack([X_train_base, X_train_feat])
            X_test_combined = np.hstack([X_test_base, X_test_feat])
            
            result = train_and_evaluate(X_train_combined, X_test_combined, y_train, y_test, group_name)
            train_time_elapsed = time.perf_counter() - train_start_time
            log.info(f"[{group_name}] Model training completed in {train_time_elapsed:.2f}s")
            
            acc = result["metrics"]["accuracy"]
            f1 = result["metrics"]["f1"]
            acc_diff = acc - baseline_acc
            f1_diff = f1 - baseline_f1
            group_time = time.perf_counter() - group_start
            
            log.info(f"\n[{group_name}] Results (completed in {group_time:.2f}s):")
            log.info(f"  Accuracy: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
            log.info(f"  F1: {f1:.4f} ({f1_diff:+.4f}, {f1_diff*100:+.2f}%)")
            log.info(f"  Precision: {result['metrics']['precision']:.4f}")
            log.info(f"  Recall: {result['metrics']['recall']:.4f}")
            log.info(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}")
            
            results[group_name] = {
                "metrics": result["metrics"],
                "n_features": n_features,
                "improvement": {
                    "accuracy_diff": float(acc_diff),
                    "f1_diff": float(f1_diff),
                    "accuracy_pct": float(acc_diff * 100),
                    "f1_pct": float(f1_diff * 100),
                }
            }
            
            improvements.append({
                "group": group_name,
                "accuracy_diff": acc_diff,
                "f1_diff": f1_diff,
                "accuracy": acc,
                "f1": f1,
                "n_features": n_features,
            })
        except Exception as e:
            log.error(f"[{group_name}] Failed to test: {e}")
            import traceback
            traceback.print_exc()
        
        # Periodic status update
        current_time = time.perf_counter()
        if current_time - last_status_log >= 1.0:
            elapsed_total = current_time - total_start
            remaining_groups = len(feature_groups) - group_idx
            log.info(f"[STATUS] Completed {group_idx}/{len(feature_groups)} groups in {elapsed_total:.0f}s - {remaining_groups} remaining")
            last_status_log = current_time
    
    total_time = time.perf_counter() - total_start
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 3 COMPLETE: Tested {len(improvements)} feature groups in {total_time:.2f}s")
    log.info(f"{'='*80}")
    
    # Sort by improvement
    improvements.sort(key=lambda x: x["accuracy_diff"], reverse=True)
    
    # Report results
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    log.info("\nFeature groups ranked by accuracy improvement:")
    log.info(f"{'Group':<25} {'Features':<10} {'Accuracy':<12} {'Acc Diff':<12} {'F1 Diff':<12} {'Status':<10}")
    log.info("-" * 80)
    
    for imp in improvements:
        acc_diff = imp["accuracy_diff"]
        f1_diff = imp["f1_diff"]
        status = "✅ HELPS" if acc_diff > 0.005 else ("⚠️  NEUTRAL" if acc_diff > -0.005 else "❌ HURTS")
        log.info(f"{imp['group']:<25} {imp['n_features']:<10} {imp['accuracy']:<12.4f} {acc_diff:+12.4f} {f1_diff:+12.4f} {status:<10}")
    
    # Identify helpful groups
    helpful_groups = [imp for imp in improvements if imp["accuracy_diff"] > 0.005]
    
    # Test combination of helpful groups
    if helpful_groups:
        log.info(f"\n{'='*80}")
        log.info("PHASE 4: Testing combination of helpful feature groups")
        log.info(f"{'='*80}")
        
        helpful_group_names = [imp["group"] for imp in helpful_groups]
        log.info(f"Combining {len(helpful_group_names)} helpful groups: {', '.join(helpful_group_names)}")
        combined_start = time.perf_counter()
        
        # Extract all helpful features
        log.info("Extracting all helpful features...")
        X_train_helpful = X_train_base.copy()
        X_test_helpful = X_test_base.copy()
        
        for group_idx, group_name in enumerate(helpful_group_names, 1):
            log.info(f"  [{group_idx}/{len(helpful_group_names)}] Extracting {group_name}...")
            extract_fn, _ = feature_groups[group_name]
            X_feat = extract_fn(paths)
            X_train_helpful = np.hstack([X_train_helpful, X_feat[train_indices]])
            X_test_helpful = np.hstack([X_test_helpful, X_feat[test_indices]])
        
        log.info(f"Training combined model with {X_train_helpful.shape[1]} total features...")
        result_combined = train_and_evaluate(X_train_helpful, X_test_helpful, y_train, y_test, "Combined helpful")
        combined_time = time.perf_counter() - combined_start
        
        combined_acc = result_combined["metrics"]["accuracy"]
        combined_f1 = result_combined["metrics"]["f1"]
        combined_acc_diff = combined_acc - baseline_acc
        combined_f1_diff = combined_f1 - baseline_f1
        
        log.info(f"\nCombined helpful features results (completed in {combined_time:.2f}s):")
        log.info(f"  Accuracy: {combined_acc:.4f} ({combined_acc_diff:+.4f}, {combined_acc_diff*100:+.2f}%)")
        log.info(f"  F1: {combined_f1:.4f} ({combined_f1_diff:+.4f}, {combined_f1_diff*100:+.2f}%)")
        log.info(f"  Precision: {result_combined['metrics']['precision']:.4f}")
        log.info(f"  Recall: {result_combined['metrics']['recall']:.4f}")
        log.info(f"  ROC-AUC: {result_combined['metrics']['roc_auc']:.4f}")
        
        results["combined_helpful"] = {
            "groups": helpful_group_names,
            "metrics": result_combined["metrics"],
            "improvement": {
                "accuracy_diff": float(combined_acc_diff),
                "f1_diff": float(combined_f1_diff),
            }
        }
    
    # Save results
    log.info(f"\n{'='*80}")
    log.info("PHASE 5: Saving results")
    log.info(f"{'='*80}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline": results_baseline,
            "feature_groups": results,
            "ranked_improvements": improvements,
            "helpful_groups": [imp["group"] for imp in helpful_groups],
        }, f, indent=2)
    
    log.info(f"Results saved to {args.output}")
    
    total_script_time = time.perf_counter() - dataset_start
    log.info(f"\n{'='*80}")
    log.info(f"FEATURE DISCOVERY COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total time: {total_script_time:.2f}s ({total_script_time/60:.1f} minutes)")
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    if helpful_groups:
        best_improvement = improvements[0]
        log.info(f"Best improvement: {best_improvement['group']} (+{best_improvement['accuracy_diff']*100:.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

