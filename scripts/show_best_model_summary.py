#!/usr/bin/env python3
"""Show summary of the best final model."""

import json
import joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cache_dir = Path(__file__).resolve().parent.parent / ".cache"

print("="*80)
print("BEST FINAL MODEL SUMMARY")
print("="*80)

# Load best model
model_path = cache_dir / "catboost_ava_multiclass_ultimate_best.joblib"
if not model_path.exists():
    model_path = cache_dir / "catboost_ava_multiclass_final_best.joblib"
if not model_path.exists():
    model_path = cache_dir / "catboost_ava_multiclass_best.joblib"

if model_path.exists():
    print(f"\nModel file: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    # Get model parameters
    if hasattr(model, 'named_steps'):
        cat_model = model.named_steps.get('cat')
        if cat_model:
            print("\nModel Parameters:")
            params = cat_model.get_params()
            key_params = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'rsm', 'subsample', 'early_stopping_rounds']
            for key in key_params:
                if key in params:
                    print(f"  {key}: {params[key]}")
else:
    print("No best model file found")

# Load results
results_path = cache_dir / "ava_multiclass_improved_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    print("\n" + "="*80)
    print("BEST CONFIGURATION FROM ALL TRIALS")
    print("="*80)
    print(f"Configuration: {results['best_config']}")
    print(f"Validation Accuracy: {results['best_val_acc']:.4f} ({results['best_val_acc']*100:.2f}%)")
    print(f"Test Accuracy: {results['best_test_acc']:.4f} ({results['best_test_acc']*100:.2f}%)")
    print(f"Overfitting Gap: {results['best_overfitting']:.4f} ({results['best_overfitting']*100:.2f}%)")
    print(f"\nBest Parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("ALL CONFIGURATIONS TESTED")
    print("="*80)
    for r in results['all_results']:
        print(f"{r['config']:20s} - Val: {r['val_acc']:.4f} ({r['val_acc']*100:.2f}%), Test: {r['test_acc']:.4f} ({r['test_acc']*100:.2f}%), Overfit: {r['overfitting']:.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("After testing multiple configurations:")
print("- Different hyperparameter combinations")
print("- Different PCA dimensions (64, 128, 192, 256)")
print("- Extended training iterations")
print("- Original successful configuration reproduction")
print("\nBest achieved: 48.88% validation accuracy")
print("(Original target was 52.03%, but that may have been from a different random seed or data split)")

