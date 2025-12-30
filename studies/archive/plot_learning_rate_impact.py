#!/usr/bin/env python3
"""Plot test accuracy vs learning rate."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("plot_lr")


def train_and_evaluate_cv(X, y, learning_rate: float, iterations: int = 200, cv_folds: int = 5) -> tuple[float, float]:
    """Train model with cross-validation and return mean accuracy and std."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=6,
            l2_leaf_reg=1.0,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )),
    ])
    
    # Use stratified K-fold for cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    return float(np.mean(scores)), float(np.std(scores))


def main():
    # Load data
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    repo = RatingsTagsRepository(path=os.path.join(image_dir, ".ratings_tags.json"))
    
    X, y, _ = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Data: {len(y)} total samples")
    
    # Test learning rates with cross-validation
    learning_rates = np.linspace(0.01, 0.3, 30)  # 30 points for smooth curve
    
    print("Testing learning rates with 5-fold cross-validation...")
    accuracies = []
    acc_stds = []
    for lr in learning_rates:
        mean_acc, std_acc = train_and_evaluate_cv(X, y, learning_rate=lr, cv_folds=5)
        accuracies.append(mean_acc)
        acc_stds.append(std_acc)
        print(f"LR={lr:.3f}: {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        # Plot with error bars
        plt.errorbar(learning_rates, accuracies, yerr=acc_stds, fmt='b-o', linewidth=2, 
                    markersize=4, capsize=3, capthick=1.5, alpha=0.7, label='CV Accuracy ± 1 std')
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Cross-Validation Accuracy', fontsize=12)
        plt.title('CV Accuracy vs Learning Rate (5-fold Cross-Validation)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = np.argmax(accuracies)
        best_lr = learning_rates[best_idx]
        best_acc = accuracies[best_idx]
        best_std = acc_stds[best_idx]
        plt.plot(best_lr, best_acc, 'ro', markersize=10, 
                label=f'Best: LR={best_lr:.3f}, Acc={best_acc:.4f}±{best_std:.4f}')
        
        # Mark current production (0.1)
        current_lr = 0.1
        if current_lr in learning_rates:
            current_idx = np.where(learning_rates == current_lr)[0][0]
            current_acc = accuracies[current_idx]
            current_std = acc_stds[current_idx]
            plt.plot(current_lr, current_acc, 'go', markersize=10, 
                    label=f'Current: LR={current_lr:.1f}, Acc={current_acc:.4f}±{current_std:.4f}')
        else:
            # Interpolate
            current_acc = np.interp(current_lr, learning_rates, accuracies)
            current_std = np.interp(current_lr, learning_rates, acc_stds)
            plt.plot(current_lr, current_acc, 'go', markersize=10, 
                    label=f'Current: LR={current_lr:.1f}, Acc={current_acc:.4f}±{current_std:.4f}')
        
        # Mark new production (0.07)
        new_lr = 0.07
        if new_lr in learning_rates:
            new_idx = np.where(learning_rates == new_lr)[0][0]
            new_acc = accuracies[new_idx]
            new_std = acc_stds[new_idx]
            plt.plot(new_lr, new_acc, 'mo', markersize=10, 
                    label=f'New: LR={new_lr:.2f}, Acc={new_acc:.4f}±{new_std:.4f}')
        else:
            new_acc = np.interp(new_lr, learning_rates, accuracies)
            new_std = np.interp(new_lr, learning_rates, acc_stds)
            plt.plot(new_lr, new_acc, 'mo', markersize=10, 
                    label=f'New: LR={new_lr:.2f}, Acc={new_acc:.4f}±{new_std:.4f}')
        
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save plot
        output_path = Path(__file__).parent.parent / "studies" / "outputs" / "learning_rate_impact_cv.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        # Also show summary
        print(f"\nSummary (5-fold CV):")
        print(f"  Best LR: {best_lr:.3f} (Accuracy: {best_acc:.4f}±{best_std:.4f} = {best_acc*100:.2f}%±{best_std*100:.2f}%)")
        print(f"  Current LR (0.1): {current_acc:.4f}±{current_std:.4f} = {current_acc*100:.2f}%±{current_std*100:.2f}%")
        print(f"  New LR (0.07): {new_acc:.4f}±{new_std:.4f} = {new_acc*100:.2f}%±{new_std*100:.2f}%")
        if current_lr in learning_rates or new_lr in learning_rates:
            print(f"  Improvement: {new_acc - current_acc:+.4f} ({(new_acc - current_acc)*100:+.2f} pp)")
        
    except ImportError:
        print("\nMatplotlib not available. Showing data only:")
        print(f"\n{'LR':<10} {'Accuracy':<12}")
        print("-" * 25)
        for lr, acc in zip(learning_rates, accuracies):
            marker = " ← BEST" if acc == max(accuracies) else ""
            print(f"{lr:<10.3f} {acc:<12.4f} ({acc*100:.2f}%){marker}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

