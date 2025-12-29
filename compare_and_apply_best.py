#!/usr/bin/env python3
"""Compare all tuning results and apply the best parameters."""
import json
import sys
from pathlib import Path

# Best-effort early mitigation when running CLI tools that may import native libs
try:
    import src.init_mitigation  # noqa: F401
except Exception:
    pass


def main():
    print("=" * 80)
    print("COMPARING ALL AVAILABLE TUNING RESULTS")
    print("=" * 80)
    print()

    results = []

    # 1. Fine-tuning results (Nov 2, 2025)
    try:
        with open('sweeps/fine_tuning_results.json') as f:
            fine_results = json.load(f)
        best_fine = max(fine_results, key=lambda x: x['mean_accuracy'])
        results.append({
            'source': 'Fine-Tuning (Nov 2, 2025)',
            'file': 'sweeps/fine_tuning_results.json',
            'accuracy': best_fine['mean_accuracy'],
            'std': best_fine['std_accuracy'],
            'auc': best_fine['mean_auc'],
            'f1': best_fine['mean_f1'],
            'folds': best_fine['folds'],
            'params': best_fine['params']
        })
        print(f"✓ Fine-Tuning: {best_fine['mean_accuracy']:.4f} accuracy (100 trials)")
    except Exception as e:
        print(f"✗ Fine-Tuning: {e}")

    # 2. Combined search (Nov 1, 2025)
    try:
        with open('sweeps/combined_search_results.json') as f:
            combined_results = json.load(f)
        best_combined = max(combined_results, key=lambda x: x['mean'])
        results.append({
            'source': 'Combined Search (Nov 1, 2025)',
            'file': 'sweeps/combined_search_results.json',
            'accuracy': best_combined['mean'],
            'std': best_combined['std'],
            'auc': 0,  # Not recorded
            'f1': 0,   # Not recorded
            'folds': 3,
            'params': best_combined['params']
        })
        print(f"✓ Combined Search: {best_combined['mean']:.4f} accuracy (~100 trials)")
    except Exception as e:
        print(f"✗ Combined Search: {e}")

    # 3. Currently applied
    try:
        with open(Path.home() / '.photo-derush-xgb-params.json') as f:
            current_params = json.load(f)
        print(f"✓ Currently Applied: {len(current_params)} parameters")
    except Exception as e:
        print(f"✗ Currently Applied: {e}")
        current_params = None

    print()
    print("=" * 80)
    print("RANKING")
    print("=" * 80)

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['source']}")
        print(f"   Accuracy:  {r['accuracy']:.4f} ± {r['std']:.4f} ({r['folds']}-fold CV)")
        if r['auc'] > 0:
            print(f"   ROC AUC:   {r['auc']:.4f}")
        if r['f1'] > 0:
            print(f"   F1 Score:  {r['f1']:.4f}")

    if not results:
        print("No results found!")
        return 1

    # Best configuration
    best = results[0]

    print()
    print("=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"\nSource: {best['source']}")
    print(f"Accuracy: {best['accuracy']:.4f} ± {best['std']:.4f}")
    if best['auc'] > 0:
        print(f"ROC AUC:  {best['auc']:.4f}")
    if best['f1'] > 0:
        print(f"F1 Score: {best['f1']:.4f}")

    print("\nParameters:")
    for k, v in sorted(best['params'].items()):
        print(f"  {k:20s} = {v}")

    # Check if we need to apply
    needs_update = True
    if current_params:
        # Compare (ignore _scale_pos_weight_ratio)
        current_clean = {k: v for k, v in current_params.items() if not k.startswith('_')}
        best_clean = {k: v for k, v in best['params'].items() if not k.startswith('_')}

        if current_clean == best_clean:
            needs_update = False

    print()
    print("=" * 80)

    if needs_update:
        print("⚠  APPLYING BEST PARAMETERS...")
        print("=" * 80)

        # Import and save
        sys.path.insert(0, str(Path(__file__).parent))
        from src.tuning import save_best_params

        save_best_params(best['params'])
        print("\n✓ Applied to: ~/.photo-derush-xgb-params.json")
        print(f"  Improvement: +{(best['accuracy'] - (results[1]['accuracy'] if len(results) > 1 else 0)) * 100:.2f} percentage points over previous best")
    else:
        print("✓ ALREADY USING BEST PARAMETERS")
        print("=" * 80)
        print("\nNo action needed - you already have the optimal configuration!")

    print()
    return 0

if __name__ == '__main__':
    sys.exit(main())

