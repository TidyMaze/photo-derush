#!/usr/bin/env python3
"""List all features with influence scores from the app model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_model
from src.model_stats import _get_feature_names
from src.training_core import DEFAULT_MODEL_PATH

def main():
    # Load the app model
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return 1
    
    print(f"Loading model from {model_path}...")
    loaded = load_model(model_path)
    if loaded is None:
        print("Failed to load model")
        return 1
    
    model = loaded.model
    meta = loaded.meta
    
    # Get feature importances
    try:
        xgb_model = model.named_steps.get("xgb")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
        else:
            print("Model does not have feature importances")
            return 1
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        return 1
    
    # Get feature names
    feature_names = _get_feature_names()
    
    # Sort by importance (descending)
    indices_sorted = np.argsort(importances)[::-1]
    
    # Calculate total importance for percentage
    total_importance = np.sum(importances)
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS (App Model)")
    print("="*80)
    print(f"\nModel path: {model_path}")
    print(f"Total features: {len(importances)}")
    print(f"Model samples: {meta.get('n_samples', 'unknown')}")
    print(f"Model precision: {meta.get('precision', 0.0):.4f}")
    print(f"Model ROC-AUC: {meta.get('roc_auc', 0.0):.4f}")
    print(f"Model F1: {meta.get('f1', 0.0):.4f}\n")
    
    print(f"{'Rank':<6} {'Feature Index':<15} {'Feature Name':<40} {'Importance':<12} {'%':<8}")
    print("-" * 80)
    
    for rank, idx in enumerate(indices_sorted, 1):
        name = feature_names.get(idx, f"feature_{idx}")
        importance = importances[idx]
        percentage = (importance / total_importance * 100) if total_importance > 0 else 0
        print(f"{rank:<6} {idx:<15} {name:<40} {importance:<12.6f} {percentage:<8.2f}%")
    
    print("\n" + "="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())



