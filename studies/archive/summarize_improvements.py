#!/usr/bin/env python3
"""Summarize all improvement results.

Usage:
    poetry run python scripts/summarize_improvements.py
"""

from __future__ import annotations

import json
from pathlib import Path

def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    print("="*80)
    print("IMPROVEMENT PLAN RESULTS SUMMARY")
    print("="*80)
    print()
    
    baseline_acc = 0.72
    
    # 1. Threshold optimization
    threshold_path = cache_dir / "threshold_optimization_results.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
            print("1. THRESHOLD OPTIMIZATION ✅")
            print(f"   Optimal threshold: {data['optimal_threshold']:.3f}")
            print(f"   Test accuracy: {data['test_accuracy_optimal']:.2%} (vs {data['test_accuracy_default']:.2%} default)")
            print(f"   Improvement: +{data['improvement']*100:.2f} percentage points")
            print()
    else:
        print("1. THRESHOLD OPTIMIZATION ⏳ (not completed)")
        print()
    
    # 2. CatBoost tuning
    tuning_path = cache_dir / "catboost_tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            data = json.load(f)
            print("2. CATBOOST HYPERPARAMETER TUNING ✅")
            print(f"   Best CV accuracy: {data['best_cv_accuracy']:.2%}")
            print(f"   Best parameters: {data['best_params']}")
            print()
    else:
        print("2. CATBOOST HYPERPARAMETER TUNING ⏳ (running in background)")
        print()
    
    # 3. Embeddings
    embeddings_path = cache_dir / "embeddings_training_results.json"
    if embeddings_path.exists():
        with open(embeddings_path) as f:
            data = json.load(f)
            print("3. IMAGE EMBEDDINGS ✅")
            print(f"   Total features: {data['n_total_features']} ({data['n_handcrafted_features']} handcrafted + {data['n_embedding_features']} embeddings)")
            print(f"   Test accuracy: {data['test_metrics']['accuracy']:.2%}")
            print(f"   Improvement vs baseline: +{data['improvement_vs_baseline']:.2f} percentage points")
            print()
    else:
        print("3. IMAGE EMBEDDINGS ⏳ (embeddings building in background)")
        print()
    
    # Final summary
    print("="*80)
    print("FINAL STATUS")
    print("="*80)
    print(f"Baseline accuracy: {baseline_acc:.2%}")
    
    current_best = baseline_acc
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
            current_best = max(current_best, data['test_accuracy_optimal'])
    
    print(f"Current best: {current_best:.2%} (threshold optimization)")
    print(f"Total improvement so far: +{(current_best - baseline_acc)*100:.2f} percentage points")
    print()
    print("Next steps:")
    print("  - Wait for CatBoost tuning to complete")
    print("  - Wait for embeddings to build, then train with embeddings")
    print("  - Review mislabeled images (manual step)")

if __name__ == "__main__":
    main()

