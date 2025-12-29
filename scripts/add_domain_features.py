#!/usr/bin/env python3
"""Add domain-specific photography features.

Domain expertise features:
1. Composition: Rule of thirds, symmetry, subject placement
2. Technical quality: Blur detection, focus quality, noise estimation
3. Aesthetic: Color harmony, contrast balance, visual weight
4. Content: Face detection, subject size, background complexity
5. Image quality: Compression artifacts, exposure balance

Usage:
    poetry run python scripts/add_domain_features.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat, ImageFilter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

# Force unbuffered output for live progress
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
    force=True
)
log = logging.getLogger("domain_features")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    log.warning("OpenCV not available, some features will be skipped")


def detect_blur(image_path: str) -> float:
    """Detect blur using Laplacian variance - optimized."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV not available for blur detection")
    
    # Read grayscale directly - faster, downsample immediately
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Downsample aggressively for speed (blur detection works fine at low res)
    h, w = img.shape
    if h * w > 400000:  # > 0.4MP - more aggressive
        scale = (400000 / (h * w)) ** 0.5
        new_h, new_w = max(100, int(h * scale)), max(100, int(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return float(laplacian_var)


def rule_of_thirds_score(image_path: str) -> float:
    """Score based on rule of thirds composition - optimized."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    
    # Downsample more aggressively
    if w * h > 300000:  # > 0.3MP
        scale = (300000 / (w * h)) ** 0.5
        new_w, new_h = max(200, int(w * scale)), max(200, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h
    
    arr = np.array(img, dtype=np.float32)
    
    # Rule of thirds lines
    third_w = w / 3
    third_h = h / 3
    
    # Sample lines efficiently
    h1_line = arr[int(third_h), :]
    h2_line = arr[int(2 * third_h), :]
    h_variance = float(np.var(h1_line) + np.var(h2_line))
    
    v1_line = arr[:, int(third_w)]
    v2_line = arr[:, int(2 * third_w)]
    v_variance = float(np.var(v1_line) + np.var(v2_line))
    
    return (h_variance + v_variance) / (w * h)


def symmetry_score(image_path: str) -> float:
    """Measure horizontal and vertical symmetry - optimized."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    
    # Downsample more aggressively
    if w * h > 300000:  # > 0.3MP
        scale = (300000 / (w * h)) ** 0.5
        new_w, new_h = max(200, int(w * scale)), max(200, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = new_w, new_h
    
    arr = np.array(img, dtype=np.float32)
    h_arr, w_arr = arr.shape
    
    # Horizontal symmetry
    left_half = arr[:, :w_arr//2]
    right_half = np.fliplr(arr[:, w_arr//2:])[:, :w_arr//2]
    h_symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
    
    # Vertical symmetry
    top_half = arr[:h_arr//2, :]
    bottom_half = np.flipud(arr[h_arr//2:, :])[:h_arr//2, :]
    v_symmetry = 1.0 - np.mean(np.abs(top_half - bottom_half)) / 255.0
    
    return float((h_symmetry + v_symmetry) / 2.0)


def color_harmony_score(image_path: str) -> float:
    """Measure color harmony using color wheel relationships - optimized."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Downsample significantly for speed
    if w * h > 200000:  # > 0.2MP
        scale = (200000 / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    arr = np.array(img).reshape(-1, 3)
    
    # Sample fewer pixels
    sample_size = min(500, len(arr))
    indices = np.linspace(0, len(arr)-1, sample_size, dtype=int)
    sampled = arr[indices]
    
    # Convert to HSV more efficiently
    if CV2_AVAILABLE:
        sampled_2d = sampled.reshape(1, -1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(sampled_2d, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hues = hsv[:, 0]
    else:
        # Fallback to PIL
        hsv = np.array([Image.new("RGB", (1, 1), tuple(rgb)).convert("HSV").getpixel((0, 0)) for rgb in sampled])
        hues = hsv[:, 0]
    
    # Score: lower std = more harmonious
    hue_std = float(np.std(hues))
    harmony_score = 1.0 / (1.0 + hue_std / 60.0)
    
    return harmony_score


def exposure_balance(image_path: str) -> float:
    """Measure exposure balance (histogram distribution)."""
    img = Image.open(image_path).convert("L")
    hist = img.histogram()
    hist = np.array(hist)
    
    # Well-exposed images have histogram spread across range
    # Over/under exposed have peaks at extremes
    total = hist.sum()
    if total == 0:
        raise ValueError(f"Empty histogram for {image_path}")
    
    # Calculate distribution
    bins = np.arange(256)
    mean_brightness = np.sum(bins * hist) / total
    
    # Ideal exposure: mean around 128
    ideal_mean = 128.0
    balance = 1.0 - abs(mean_brightness - ideal_mean) / 128.0
    
    return float(max(0.0, balance))


def subject_size_ratio(image_path: str) -> float:
    """Estimate subject size relative to image (using edge density) - optimized."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    
    # Downsample more aggressively
    if w * h > 300000:  # > 0.3MP
        scale = (300000 / (w * h)) ** 0.5
        new_w, new_h = max(200, int(w * scale)), max(200, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Edge detection
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float32)
    
    # Edge density in center vs periphery
    h_arr, w_arr = edge_arr.shape
    center_h, center_w = h_arr // 2, w_arr // 2
    margin = min(h_arr, w_arr) // 4
    
    center_region = edge_arr[center_h-margin:center_h+margin, center_w-margin:center_w+margin]
    periphery_mask = np.ones_like(edge_arr, dtype=bool)
    periphery_mask[center_h-margin:center_h+margin, center_w-margin:center_w+margin] = False
    periphery_region = edge_arr[periphery_mask]
    
    center_density = float(np.mean(center_region > 50))
    periphery_density = float(np.mean(periphery_region > 50)) if len(periphery_region) > 0 else 0.0
    
    if periphery_density == 0:
        return 0.5
    
    ratio = center_density / (center_density + periphery_density + 1e-8)
    return float(ratio)


def extract_leading_lines(image_path: str) -> float:
    """Detect leading lines (strong directional edges) - optimized."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV not available for leading lines detection")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Downsample more aggressively
    h, w = img.shape
    if h * w > 300000:  # > 0.3MP
        scale = (300000 / (h * w)) ** 0.5
        new_h, new_w = max(200, int(h * scale)), max(200, int(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Use faster Canny params
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=15)
    if lines is None:
        return 0.0
    return float(len(lines) / 100.0)


def extract_texture_complexity(image_path: str) -> float:
    """Measure texture complexity (local variance) - optimized."""
    img = Image.open(image_path).convert("L")
    # Downsample more aggressively
    w, h = img.size
    if w * h > 300000:  # > 0.3MP
        scale = (300000 / (w * h)) ** 0.5
        new_w, new_h = max(200, int(w * scale)), max(200, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Use even larger windows and fewer samples
    h_arr, w_arr = arr.shape
    window_size = min(30, h_arr//6, w_arr//6)  # Larger windows
    if window_size < 3:
        raise ValueError(f"Image too small for texture analysis: {h_arr}x{w_arr}")
    step = window_size * 2  # Skip more windows
    local_vars = []
    for i in range(0, h_arr-window_size, step):
        for j in range(0, w_arr-window_size, step):
            window = arr[i:i+window_size, j:j+window_size]
            local_vars.append(np.var(window))
    return float(np.mean(local_vars)) if local_vars else 0.0


def extract_depth_cues(image_path: str) -> float:
    """Estimate depth cues (gradient magnitude variation) - optimized."""
    img = Image.open(image_path).convert("L")
    # Downsample more aggressively
    w, h = img.size
    if w * h > 300000:  # > 0.3MP
        scale = (300000 / (w * h)) ** 0.5
        new_w, new_h = max(200, int(w * scale)), max(200, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Use simpler gradient calculation (skip sqrt for speed, use abs)
    grad_y = np.diff(arr, axis=0)
    grad_x = np.diff(arr, axis=1)
    # Use Manhattan distance approximation instead of Euclidean
    grad_mag = np.abs(grad_x[:-1, :]) + np.abs(grad_y[:, :-1])
    return float(np.std(grad_mag))


def extract_domain_features(image_paths: list[str]) -> np.ndarray:
    """Extract NEW domain-specific features (not already in base features)."""
    features = []
    
    log.info(f"\nExtracting NEW domain features for {len(image_paths)} images...")
    log.info("(This will show progress every second)")
    sys.stdout.flush()
    start_time = time.perf_counter()
    last_log_time = start_time
    
    for idx, path in enumerate(image_paths):
        current_time = time.perf_counter()
        if current_time - last_log_time >= 1.0:
            elapsed = current_time - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            pct = (idx + 1) / len(image_paths) * 100
            log.info(f"  Progress: {idx+1}/{len(image_paths)} ({pct:.1f}%) - {rate:.1f} img/s - ETA: {eta:.0f}s")
            last_log_time = current_time
            sys.stdout.flush()
            sys.stderr.flush()
        
        # NEW features not already in base feature set
        feat = [
            detect_blur(path),  # Blur detection (different from sharpness)
            extract_leading_lines(path),  # Leading lines detection
            extract_texture_complexity(path),  # Texture complexity
            extract_depth_cues(path),  # Depth perception cues
            exposure_balance(path),  # Exposure histogram balance
            subject_size_ratio(path),  # Subject size estimation
        ]
        features.append(feat)
    
    elapsed = time.perf_counter() - start_time
    log.info(f"Domain feature extraction completed in {elapsed:.2f}s")
    
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Add domain-specific features")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/domain_features_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("DOMAIN-SPECIFIC FEATURE EXTRACTION")
    log.info("="*80)
    
    # Build base dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info(f"\nLoading base dataset from {image_dir}...")
    log.info("(This may take a while - base features are being extracted/cached)")
    sys.stdout.flush()
    X_base, y, filenames = build_dataset(image_dir, repo=repo)
    X_base = np.array(X_base)
    
    log.info(f"✓ Base dataset loaded: {len(y)} samples, {X_base.shape[1]} features")
    sys.stdout.flush()
    
    # Extract domain features for all images
    image_paths = [os.path.join(image_dir, fname) for fname in filenames]
    X_domain = extract_domain_features(image_paths)
    
    log.info(f"Domain features: {X_domain.shape[1]} features")
    
    # Combine features
    X_combined = np.hstack([X_base, X_domain])
    log.info(f"Combined: {X_combined.shape[1]} total features")
    
    # Split
    indices = np.arange(len(X_combined))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X_combined[train_indices]
    X_test = X_combined[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Baseline (base features only)
    log.info("\n" + "="*80)
    log.info("BASELINE (Base Features Only)")
    log.info("="*80)
    
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    baseline_clf = Pipeline([
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
    
    X_train_base = X_base[train_indices]
    X_test_base = X_base[test_indices]
    
    baseline_clf.fit(X_train_base, y_train)
    baseline_pred = baseline_clf.predict(X_test_base)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Test with domain features
    log.info("\n" + "="*80)
    log.info("WITH DOMAIN FEATURES")
    log.info("="*80)
    log.info("Training model with domain features...")
    sys.stdout.flush()
    
    domain_clf = Pipeline([
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
    
    domain_clf.fit(X_train, y_train)
    domain_pred = domain_clf.predict(X_test)
    domain_acc = accuracy_score(y_test, domain_pred)
    
    improvement = domain_acc - baseline_acc
    
    log.info(f"With domain features: {domain_acc:.4f}")
    log.info(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        log.info("✅ Domain features improve accuracy!")
        
        # Feature importance
        importances = domain_clf.named_steps["xgb"].feature_importances_
        domain_feat_indices = list(range(X_base.shape[1], X_combined.shape[1]))
        domain_importances = importances[domain_feat_indices]
        
        feature_names = [
            "Blur Detection",
            "Rule of Thirds",
            "Symmetry",
            "Color Harmony",
            "Exposure Balance",
            "Subject Size Ratio",
        ]
        
        log.info("\nDomain feature importances:")
        for name, imp in zip(feature_names, domain_importances):
            log.info(f"  {name}: {imp:.6f}")
    else:
        log.info("❌ Domain features do not improve accuracy")
    
    # Save results
    import json
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline_accuracy": float(baseline_acc),
            "domain_features_accuracy": float(domain_acc),
            "improvement": float(improvement),
            "n_domain_features": int(X_domain.shape[1]),
            "n_total_features": int(X_combined.shape[1]),
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

