#!/usr/bin/env python3
"""Analyze gain/loss percentage for each feature individually."""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params
import xgboost as xgb


def get_feature_names():
    """Get all feature names matching the actual feature extraction order."""
    from src.features import FEATURE_COUNT
    
    names = []
    
    # 0-11: Basic stats
    names.extend([
        "Width", "Height", "Aspect Ratio", "File Size (log)",
        "Mean Brightness", "Std Brightness",
        "Mean Red", "Mean Green", "Mean Blue",
        "Std Red", "Std Green", "Std Blue",
    ])
    
    # 12-35: RGB histograms (8 bins per channel = 24 total)
    for ch, color in enumerate(["R", "G", "B"]):
        for bin in range(8):
            names.append(f"{color} Hist Bin {bin}")
    
    # 36-42: Image quality metrics
    names.extend([
        "Sharpness", "Saturation", "Entropy",
        "Std Brightness (dup)", "Highlight Clip", "Shadow Clip", "Noise Level",
    ])
    
    # 43-46: Temporal features
    names.extend(["Hour of Day", "Day of Week", "Month", "Is Weekend"])
    
    # 47-56: EXIF features
    names.extend([
        "ISO", "Aperture", "Shutter Speed", "Flash Fired",
        "Focal Length 35mm", "Digital Zoom", "Exposure Compensation",
        "White Balance", "Exposure Mode", "Metering Mode",
    ])
    
    # 57-76: Advanced photography features
    names.extend([
        "Edge Density", "Edge Strength", "Corner Count", "Face Count",
        "Histogram Balance", "Color Temperature", "Center Brightness Ratio",
        "Exposure Quality", "Color Diversity",
        "Rule of Thirds Score", "Symmetry Score", "Horizon Levelness",
        "Center Focus Quality", "Dynamic Range Utilization", "Subject Isolation",
        "Golden Hour", "Lighting Quality", "Color Harmony",
        "Sky Ground Ratio", "Motion Blur",
    ])
    
    # 77: Person Detection
    names.append("Person Detection")
    
    # Ensure we have exactly FEATURE_COUNT features
    if len(names) != FEATURE_COUNT:
        if len(names) < FEATURE_COUNT:
            names.extend([f"Unknown {i}" for i in range(FEATURE_COUNT - len(names))])
        else:
            names = names[:FEATURE_COUNT]
    
    return names


def train_model(X_train, X_test, y_train, y_test, feature_indices=None):
    """Train XGBoost model and return accuracy."""
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Select features if specified
    if feature_indices is not None:
        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]
    else:
        X_train_subset = X_train
        X_test_subset = X_test
    
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
    
    clf.fit(X_train_subset, y_train)
    y_pred = clf.predict(X_test_subset)
    return accuracy_score(y_test, y_pred)


def main():
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    # Build dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        print("No repository found")
        return 1
    
    print("Loading dataset...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    
    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    # Split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Baseline: all features
    print("\nTraining baseline model (all features)...")
    baseline_acc = train_model(X_train, X_test, y_train, y_test)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Get feature names
    feature_names = get_feature_names()
    n_features = X.shape[1]
    
    print(f"\nAnalyzing {n_features} features...")
    print("This will take a while...\n")
    
    # Test each feature removal
    results = []
    for feat_idx in range(n_features):
        name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        
        # Create feature mask without this feature
        feature_mask = [i for i in range(n_features) if i != feat_idx]
        
        # Train without this feature
        acc_without = train_model(X_train, X_test, y_train, y_test, feature_mask)
        
        # Calculate gain (positive = feature helps, negative = feature hurts)
        gain = acc_without - baseline_acc
        gain_pct = gain * 100
        
        results.append((feat_idx, name, baseline_acc, acc_without, gain, gain_pct))
        
        # Progress update
        if (feat_idx + 1) % 10 == 0:
            print(f"  Processed {feat_idx + 1}/{n_features} features...")
            sys.stdout.flush()
    
    # Sort by gain (most helpful first)
    results.sort(key=lambda x: x[4], reverse=True)
    
    print("\n" + "="*100)
    print("FEATURE GAIN ANALYSIS")
    print("="*100)
    print(f"Baseline accuracy (all features): {baseline_acc:.4f}\n")
    print(f"{'Rank':<6} {'Feature Name':<45} {'Baseline':<10} {'Without':<10} {'Gain':<12} {'Gain %':<10}")
    print("-" * 100)
    
    for rank, (feat_idx, name, baseline, acc_without, gain, gain_pct) in enumerate(results, 1):
        sign = "+" if gain >= 0 else ""
        print(f"{rank:<6} {name:<45} {baseline:<10.4f} {acc_without:<10.4f} {sign}{gain:<11.4f} {sign}{gain_pct:<9.2f}%")
    
    # Summary statistics
    positive_gains = [r for r in results if r[4] > 0]
    negative_gains = [r for r in results if r[4] < 0]
    zero_gains = [r for r in results if abs(r[4]) < 1e-6]
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Features that help (positive gain): {len(positive_gains)}")
    print(f"Features that hurt (negative gain): {len(negative_gains)}")
    print(f"Features with no effect: {len(zero_gains)}")
    
    if positive_gains:
        avg_positive = np.mean([r[4] for r in positive_gains])
        print(f"Average positive gain: {avg_positive:.4f} ({avg_positive*100:.2f}%)")
    
    if negative_gains:
        avg_negative = np.mean([r[4] for r in negative_gains])
        print(f"Average negative gain: {avg_negative:.4f} ({avg_negative*100:.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

