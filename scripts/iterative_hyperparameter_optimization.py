#!/usr/bin/env python3
"""Iterative hyperparameter optimization: optimize each parameter in turn, using best values as new base."""

import logging
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training_core import train_keep_trash_model, DEFAULT_MODEL_PATH
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def test_hyperparameter_value(
    image_dir: str,
    repo: RatingsTagsRepository,
    param_name: str,
    param_value: float | int,
    base_lr: float,
    base_iterations: int,
    base_patience: int,
    use_cv: bool = True,
):
    """Test a single hyperparameter value."""
    # Set parameters
    if param_name == "learning_rate":
        lr = param_value
        iterations = base_iterations
        patience = base_patience
    elif param_name == "max_iterations":
        lr = base_lr
        iterations = param_value
        patience = base_patience
    elif param_name == "patience":
        lr = base_lr
        iterations = base_iterations
        patience = param_value
    else:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    # Temporarily modify _build_pipeline
    from src import training_core
    original_build_pipeline = training_core._build_pipeline
    
    def modified_build_pipeline(
        random_state: int, scale_pos_weight: float, xgb_params: dict, 
        early_stopping_rounds: int | None, use_catboost: bool = True, 
        n_handcrafted_features: int = 78, fast_mode: bool = False
    ):
        """Modified _build_pipeline with custom hyperparameters."""
        if use_catboost:
            try:
                from catboost import CatBoostClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                import os
                
                cb_params = {
                    "iterations": iterations,
                    "learning_rate": lr,
                    "depth": 6,
                    "l2_leaf_reg": 1.0,
                    "scale_pos_weight": scale_pos_weight,
                    "random_seed": random_state,
                    "verbose": False,
                    "thread_count": 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1,
                }
                
                if early_stopping_rounds is not None:
                    cb_params["early_stopping_rounds"] = patience
                    cb_params["eval_metric"] = "Logloss"
                    cb_params["use_best_model"] = True
                    cb_params["verbose"] = False
                
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("cat", CatBoostClassifier(**cb_params)),
                ])
            except ImportError:
                pass
        
        return original_build_pipeline(
            random_state, scale_pos_weight, xgb_params, 
            early_stopping_rounds, use_catboost, n_handcrafted_features, fast_mode
        )
    
    training_core._build_pipeline = modified_build_pipeline
    
    try:
        result = train_keep_trash_model(
            image_dir=image_dir,
            repo=repo,
            model_path=DEFAULT_MODEL_PATH.replace(".joblib", f"_test_iter_{param_name}_{param_value}.joblib"),
            fast_mode=not use_cv,  # Use CV if use_cv=True
            early_stopping_rounds=patience,
        )
        
        if result:
            # Use CV metrics if available, otherwise test metrics
            cv_acc = result.cv_accuracy_mean or 0.0
            cv_std = result.cv_accuracy_std or 0.0
            roc_auc = result.roc_auc or 0.0
            f1 = result.f1 or 0.0
            keep_loss = result.aggregated_metrics.get('keep_loss_rate_mean') if result.aggregated_metrics else None
            
            return {
                "cv_accuracy": cv_acc,
                "cv_std": cv_std,
                "roc_auc": roc_auc,
                "f1": f1,
                "keep_loss_rate": keep_loss,
            }
        return None
    finally:
        training_core._build_pipeline = original_build_pipeline

def optimize_parameter(
    image_dir: str,
    repo: RatingsTagsRepository,
    param_name: str,
    param_values: list,
    base_lr: float,
    base_iterations: int,
    base_patience: int,
    use_cv: bool = True,
    metric: str = "cv_accuracy",
):
    """Optimize a single parameter, returning best value and score."""
    best_value = None
    best_score = -np.inf
    best_result = None
    
    print(f"\n  🔍 Optimizing {param_name}...")
    print(f"     Testing {len(param_values)} values with {'CV' if use_cv else 'single split'}")
    print(f"     Current base: LR={base_lr:.4f}, iterations={base_iterations}, patience={base_patience}")
    print()
    
    start_time = time.perf_counter()
    for i, param_value in enumerate(param_values):
        if isinstance(param_value, float):
            param_str = f"{param_value:.4f}"
        else:
            param_str = str(param_value)
        
        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining = avg_time * (len(param_values) - i - 1)
        
        print(f"    [{i+1:2d}/{len(param_values)}] {param_name}={param_str:8s} | Elapsed: {elapsed:5.1f}s | Est. remaining: {remaining:5.1f}s", end=" ... ")
        sys.stdout.flush()
        
        test_start = time.perf_counter()
        result = test_hyperparameter_value(
            image_dir, repo, param_name, param_value,
            base_lr, base_iterations, base_patience, use_cv
        )
        test_time = time.perf_counter() - test_start
        
        if result:
            # Choose metric to optimize
            if metric == "cv_accuracy":
                score = result["cv_accuracy"]
            elif metric == "roc_auc":
                score = result["roc_auc"]
            elif metric == "keep_loss_rate":
                score = -result["keep_loss_rate"] if result["keep_loss_rate"] else -1.0  # Negative because lower is better
            else:
                score = result["cv_accuracy"]
            
            is_best = score > best_score
            if is_best:
                best_score = score
                best_value = param_value
                best_result = result
                best_marker = " ⭐ NEW BEST"
            else:
                best_marker = ""
            
            if result["cv_accuracy"]:
                print(f"CV-Acc={result['cv_accuracy']:.2%}±{result['cv_std']:.2%}, ROC-AUC={result['roc_auc']:.4f}, F1={result['f1']:.4f} | {test_time:.1f}s{best_marker}")
            else:
                print(f"ROC-AUC={result['roc_auc']:.4f}, F1={result['f1']:.4f} | {test_time:.1f}s{best_marker}")
        else:
            print(f"FAILED | {test_time:.1f}s")
        
        # Clean up
        test_model_path = DEFAULT_MODEL_PATH.replace(".joblib", f"_test_iter_{param_name}_{param_value}.joblib")
        if os.path.exists(test_model_path):
            try:
                os.remove(test_model_path)
            except:
                pass
    
    total_time = time.perf_counter() - start_time
    print(f"\n  ✅ {param_name} optimization complete in {total_time:.1f}s")
    print(f"     Best value: {best_value} (score: {best_score:.4f})")
    
    return best_value, best_result

def iterative_optimization(
    image_dir: str,
    repo: RatingsTagsRepository,
    initial_lr: float = 0.05,
    initial_iterations: int = 1000,
    initial_patience: int = 100,
    max_iterations: int = 5,
    use_cv: bool = True,
    metric: str = "cv_accuracy",
):
    """Iteratively optimize hyperparameters."""
    print("=" * 80)
    print("ITERATIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"\nInitial parameters:")
    print(f"  Learning Rate: {initial_lr}")
    print(f"  Max Iterations: {initial_iterations}")
    print(f"  Patience: {initial_patience}")
    print(f"  Using CV: {use_cv}")
    print(f"  Optimizing for: {metric}")
    print(f"  Max iterations: {max_iterations}")
    print()
    
    # Define search ranges
    lr_values = np.logspace(np.log10(0.01), np.log10(0.3), 15)  # Focused range around optimal
    iter_values = np.linspace(50, 1000, 15, dtype=int)
    pat_values = np.linspace(10, 150, 15, dtype=int)
    
    current_lr = initial_lr
    current_iterations = initial_iterations
    current_patience = initial_patience
    
    history = []
    convergence_count = 0  # Track consecutive iterations with no significant change
    required_convergence = 2  # Require 2 consecutive iterations of convergence
    
    for iteration in range(max_iterations):
        print("\n" + "=" * 80)
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print("=" * 80)
        print(f"\nCurrent parameters: LR={current_lr:.4f}, iterations={current_iterations}, patience={current_patience}")
        print()
        
        # Optimize Learning Rate
        print(f"\n{'='*60}")
        print(f"  STEP 1/3: Optimizing Learning Rate")
        print(f"{'='*60}")
        best_lr, lr_result = optimize_parameter(
            image_dir, repo, "learning_rate", lr_values,
            current_lr, current_iterations, current_patience, use_cv, metric
        )
        print(f"\n  📊 Learning Rate Results:")
        print(f"     Best LR: {best_lr:.4f} (was {current_lr:.4f})")
        print(f"     CV-Acc: {lr_result['cv_accuracy']:.2%}±{lr_result['cv_std']:.2%}")
        print(f"     ROC-AUC: {lr_result['roc_auc']:.4f}")
        print(f"     F1: {lr_result['f1']:.4f}")
        if lr_result.get('keep_loss_rate'):
            print(f"     Keep-Loss: {lr_result['keep_loss_rate']:.2%}")
        current_lr = best_lr
        
        # Optimize Max Iterations
        print(f"\n{'='*60}")
        print(f"  STEP 2/3: Optimizing Max Iterations")
        print(f"{'='*60}")
        best_iter, iter_result = optimize_parameter(
            image_dir, repo, "max_iterations", iter_values,
            current_lr, current_iterations, current_patience, use_cv, metric
        )
        print(f"\n  📊 Max Iterations Results:")
        print(f"     Best Iterations: {best_iter} (was {current_iterations})")
        print(f"     CV-Acc: {iter_result['cv_accuracy']:.2%}±{iter_result['cv_std']:.2%}")
        print(f"     ROC-AUC: {iter_result['roc_auc']:.4f}")
        print(f"     F1: {iter_result['f1']:.4f}")
        if iter_result.get('keep_loss_rate'):
            print(f"     Keep-Loss: {iter_result['keep_loss_rate']:.2%}")
        current_iterations = best_iter
        
        # Optimize Patience
        print(f"\n{'='*60}")
        print(f"  STEP 3/3: Optimizing Patience")
        print(f"{'='*60}")
        best_pat, pat_result = optimize_parameter(
            image_dir, repo, "patience", pat_values,
            current_lr, current_iterations, current_patience, use_cv, metric
        )
        print(f"\n  📊 Patience Results:")
        print(f"     Best Patience: {best_pat} (was {current_patience})")
        print(f"     CV-Acc: {pat_result['cv_accuracy']:.2%}±{pat_result['cv_std']:.2%}")
        print(f"     ROC-AUC: {pat_result['roc_auc']:.4f}")
        print(f"     F1: {pat_result['f1']:.4f}")
        if pat_result.get('keep_loss_rate'):
            print(f"     Keep-Loss: {pat_result['keep_loss_rate']:.2%}")
        current_patience = best_pat
        
        # Record history
        history.append({
            "iteration": iteration + 1,
            "lr": current_lr,
            "iterations": current_iterations,
            "patience": current_patience,
            "cv_accuracy": pat_result['cv_accuracy'],
            "cv_std": pat_result['cv_std'],
            "roc_auc": pat_result['roc_auc'],
            "f1": pat_result['f1'],
            "keep_loss_rate": pat_result.get('keep_loss_rate'),
        })
        
        # Check convergence (if parameters didn't change much, stop early)
        if iteration > 0:
            prev = history[-2]
            lr_change = abs(current_lr - prev["lr"]) / prev["lr"] if prev["lr"] > 0 else 1.0
            iter_change = abs(current_iterations - prev["iterations"]) / prev["iterations"] if prev["iterations"] > 0 else 1.0
            pat_change = abs(current_patience - prev["patience"]) / prev["patience"] if prev["patience"] > 0 else 1.0
            
            print(f"\n  📈 Parameter Changes:")
            print(f"     LR: {prev['lr']:.4f} → {current_lr:.4f} ({lr_change*100:.1f}% change)")
            print(f"     Iterations: {prev['iterations']} → {current_iterations} ({iter_change*100:.1f}% change)")
            print(f"     Patience: {prev['patience']} → {current_patience} ({pat_change*100:.1f}% change)")
            print(f"     CV-Acc improvement: {prev['cv_accuracy']:.2%} → {pat_result['cv_accuracy']:.2%} ({(pat_result['cv_accuracy']-prev['cv_accuracy'])*100:+.2f}%)")
            
            if lr_change < 0.05 and iter_change < 0.05 and pat_change < 0.05:
                convergence_count += 1
                print(f"\n  ✅ Convergence detected! All parameters changed < 5%")
                print(f"     Convergence count: {convergence_count}/{required_convergence}")
                if convergence_count >= required_convergence:
                    print(f"     Stable optimum reached! Stopping optimization.")
                    break
                else:
                    print(f"     Continuing to verify stability...")
            else:
                convergence_count = 0  # Reset if parameters changed significantly
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal optimal parameters:")
    print(f"  Learning Rate: {current_lr:.4f}")
    print(f"  Max Iterations: {current_iterations}")
    print(f"  Patience: {current_patience}")
    print(f"\nFinal performance:")
    final_result = history[-1]
    print(f"  CV Accuracy: {final_result['cv_accuracy']:.2%}±{final_result['cv_std']:.2%}")
    print(f"  ROC-AUC: {final_result['roc_auc']:.4f}")
    print(f"  F1 Score: {final_result['f1']:.4f}")
    if final_result.get('keep_loss_rate'):
        print(f"  Keep-Loss Rate: {final_result['keep_loss_rate']:.2%}")
    
    return current_lr, current_iterations, current_patience, history

def plot_optimization_history(history: list, output_path: str = "plots/iterative_optimization_history.png"):
    """Plot the optimization history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = [h["iteration"] for h in history]
    lrs = [h["lr"] for h in history]
    iters = [h["iterations"] for h in history]
    pats = [h["patience"] for h in history]
    cv_accs = [h["cv_accuracy"] for h in history]
    cv_stds = [h["cv_std"] for h in history]
    roc_aucs = [h["roc_auc"] for h in history]
    
    # Parameter evolution
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(iterations, lrs, 'o-', label='Learning Rate', linewidth=2, markersize=8)
    ax1_twin.plot(iterations, iters, 's-', color='green', label='Max Iterations', linewidth=2, markersize=8)
    ax1_twin.plot(iterations, pats, '^-', color='red', label='Patience', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Learning Rate', fontsize=11, color='blue')
    ax1_twin.set_ylabel('Iterations / Patience', fontsize=11)
    ax1.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # CV Accuracy evolution
    ax2 = axes[0, 1]
    ax2.errorbar(iterations, cv_accs, yerr=cv_stds, fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('CV Accuracy', fontsize=11)
    ax2.set_title('CV Accuracy Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.70, 0.80])
    
    # ROC-AUC evolution
    ax3 = axes[1, 0]
    ax3.plot(iterations, roc_aucs, 'o-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('ROC-AUC', fontsize=11)
    ax3.set_title('ROC-AUC Evolution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.80, 0.90])
    
    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = []
    for h in history:
        table_data.append([
            f"{h['iteration']}",
            f"{h['lr']:.4f}",
            f"{h['iterations']}",
            f"{h['patience']}",
            f"{h['cv_accuracy']:.2%}",
            f"{h['roc_auc']:.4f}",
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Iter', 'LR', 'Iterations', 'Patience', 'CV Acc', 'ROC-AUC'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Optimization History', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nOptimization history plot saved: {output_path}")
    plt.close()

def main():
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return
    
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    if not os.path.exists(repo_path):
        logging.error(f"Repository file not found: {repo_path}")
        return
    
    repo = RatingsTagsRepository(repo_path)
    
    # Run iterative optimization
    final_lr, final_iterations, final_patience, history = iterative_optimization(
        image_dir, repo,
        initial_lr=0.05,
        initial_iterations=1000,
        initial_patience=100,
        max_iterations=5,
        use_cv=True,  # Use CV for robust evaluation
        metric="cv_accuracy",  # Optimize for CV accuracy
    )
    
    # Plot history
    plot_optimization_history(history)
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nOptimal Fast Mode Settings:")
    print(f"  learning_rate = {final_lr:.4f}")
    print(f"  max_iterations = {final_iterations}")
    print(f"  patience = {final_patience}")
    print(f"\nUpdate src/training_core.py with these values for fast_mode=True")

if __name__ == "__main__":
    main()

