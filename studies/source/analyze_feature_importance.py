#!/usr/bin/env python3
"""Analyze feature importance for all features."""

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
    
    # Train model
    print("\nTraining XGBoost model...")
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
    
    # Get feature importances
    xgb_model = clf.named_steps['xgb']
    importances = xgb_model.feature_importances_
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Sort by importance
    indices_sorted = np.argsort(importances)[::-1]
    
    # Calculate total importance for percentage
    total_importance = np.sum(importances)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    print(f"\nTotal features: {len(feature_names)}")
    print(f"Model accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}\n")
    
    print(f"{'Rank':<6} {'Feature Name':<40} {'Importance':<12} {'%':<8}")
    print("-" * 80)
    
    for rank, idx in enumerate(indices_sorted, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        importance = importances[idx]
        percentage = (importance / total_importance * 100) if total_importance > 0 else 0
        print(f"{rank:<6} {name:<40} {importance:<12.6f} {percentage:<8.2f}%")
    
    # Group by feature type
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE BY CATEGORY")
    print("="*80)
    
    categories = {
        "Basic Properties": list(range(0, 12)),
        "Histogram": list(range(12, 28)),
        "EXIF": list(range(28, 48)),
        "Advanced Photography": list(range(48, 78)),
    }
    
    for cat_name, indices_list in categories.items():
        cat_importance = sum(importances[i] for i in indices_list if i < len(importances))
        cat_percentage = (cat_importance / total_importance * 100) if total_importance > 0 else 0
        print(f"{cat_name:<30} {cat_importance:.6f} ({cat_percentage:.2f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

