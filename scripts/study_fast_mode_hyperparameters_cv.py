#!/usr/bin/env python3
"""Study the independent impact of learning rate, max iterations, and patience WITH CV enabled."""

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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")  # Reduce noise

def test_hyperparameter(
    image_dir: str,
    repo: RatingsTagsRepository,
    param_name: str,
    param_values: list,
    base_lr: float = 0.05,
    base_iterations: int = 1000,
    base_patience: int = 100,
):
    """Test a single hyperparameter with other params fixed. Uses CV (fast_mode=False)."""
    results = []
    
    print(f"\nTesting {param_name}...")
    print(f"Base settings: LR={base_lr}, iterations={base_iterations}, patience={base_patience}")
    print(f"Testing {len(param_values)} values: {param_values[:3]}...{param_values[-3:]}")
    print("Using CV (5-fold) for robust evaluation")
    print()
    
    for i, param_value in enumerate(param_values):
        # Set parameters based on which one we're testing
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
        
        if isinstance(param_value, float):
            param_str = f"{param_value:.3f}"
        else:
            param_str = str(param_value)
        print(f"[{i+1}/{len(param_values)}] {param_name}={param_str}", end=" ... ")
        
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
            t_start = time.perf_counter()
            # Use fast_mode=False to enable CV
            result = train_keep_trash_model(
                image_dir=image_dir,
                repo=repo,
                model_path=DEFAULT_MODEL_PATH.replace(".joblib", f"_test_{param_name}_{i}.joblib"),
                fast_mode=False,  # Enable CV for robust evaluation
                early_stopping_rounds=patience,
            )
            t_time = time.perf_counter() - t_start
            
            if result:
                # Use CV metrics if available, otherwise fall back to test metrics
                cv_acc = result.cv_accuracy_mean or 0.0
                cv_std = result.cv_accuracy_std or 0.0
                roc_auc = result.roc_auc or 0.0
                f1 = result.f1 or 0.0
                
                # Get keep-loss rate from aggregated_metrics if available
                keep_loss = None
                if result.aggregated_metrics:
                    keep_loss = result.aggregated_metrics.get('keep_loss_rate_mean')
                
                results.append({
                    "param_value": param_value,
                    "cv_accuracy": cv_acc,
                    "cv_std": cv_std,
                    "roc_auc": roc_auc,
                    "f1": f1,
                    "keep_loss_rate": keep_loss,
                    "training_time": t_time,
                })
                if keep_loss:
                    print(f"CV-Acc={cv_acc:.2%}±{cv_std:.2%}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, Keep-Loss={keep_loss:.2%}, time={t_time:.1f}s")
                else:
                    print(f"CV-Acc={cv_acc:.2%}±{cv_std:.2%}, ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, time={t_time:.1f}s")
            else:
                print("FAILED")
                results.append({
                    "param_value": param_value,
                    "cv_accuracy": 0.0,
                    "cv_std": 0.0,
                    "roc_auc": 0.0,
                    "f1": 0.0,
                    "keep_loss_rate": None,
                    "training_time": 0.0,
                })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "param_value": param_value,
                "cv_accuracy": 0.0,
                "cv_std": 0.0,
                "roc_auc": 0.0,
                "f1": 0.0,
                "keep_loss_rate": None,
                "training_time": 0.0,
            })
        finally:
            training_core._build_pipeline = original_build_pipeline
        
        # Clean up test model
        test_model_path = DEFAULT_MODEL_PATH.replace(".joblib", f"_test_{param_name}_{i}.joblib")
        if os.path.exists(test_model_path):
            try:
                os.remove(test_model_path)
            except:
                pass
    
    return results

def plot_results(param_name: str, results: list, output_dir: str = "plots"):
    """Plot results for a hyperparameter."""
    os.makedirs(output_dir, exist_ok=True)
    
    param_values = [r["param_value"] for r in results]
    cv_accs = [r["cv_accuracy"] for r in results]
    cv_stds = [r["cv_std"] for r in results]
    roc_aucs = [r["roc_auc"] for r in results]
    f1_scores = [r["f1"] for r in results]
    keep_loss_rates = [r["keep_loss_rate"] if r["keep_loss_rate"] is not None else 0.0 for r in results]
    training_times = [r["training_time"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # CV Accuracy plot with error bars
    axes[0, 0].errorbar(param_values, cv_accs, yerr=cv_stds, fmt='o-', linewidth=1.5, markersize=4, capsize=3)
    axes[0, 0].set_xlabel(param_name.replace('_', ' ').title())
    axes[0, 0].set_ylabel('CV Accuracy')
    axes[0, 0].set_title(f'CV Accuracy vs {param_name.replace("_", " ").title()} (with std)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    if param_name == "learning_rate":
        axes[0, 0].set_xscale('log')
    
    # ROC-AUC plot
    axes[0, 1].plot(param_values, roc_aucs, 'o-', color='green', linewidth=1.5, markersize=4)
    axes[0, 1].set_xlabel(param_name.replace('_', ' ').title())
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title(f'ROC-AUC vs {param_name.replace("_", " ").title()}')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    if param_name == "learning_rate":
        axes[0, 1].set_xscale('log')
    
    # Keep-Loss Rate plot (lower is better)
    axes[1, 0].plot(param_values, keep_loss_rates, 'o-', color='red', linewidth=1.5, markersize=4)
    axes[1, 0].set_xlabel(param_name.replace('_', ' ').title())
    axes[1, 0].set_ylabel('Keep-Loss Rate')
    axes[1, 0].set_title(f'Keep-Loss Rate vs {param_name.replace("_", " ").title()} (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.02, color='orange', linestyle='--', label='2% target')
    axes[1, 0].legend()
    if param_name == "learning_rate":
        axes[1, 0].set_xscale('log')
    
    # Training time plot
    axes[1, 1].plot(param_values, training_times, 'o-', color='purple', linewidth=1.5, markersize=4)
    axes[1, 1].set_xlabel(param_name.replace('_', ' ').title())
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].set_title(f'Training Time vs {param_name.replace("_", " ").title()}')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=5.0, color='orange', linestyle='--', label='5s target')
    axes[1, 1].legend()
    if param_name == "learning_rate":
        axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"fast_mode_{param_name}_study_cv.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()
    
    # Find best values
    best_cv_idx = np.argmax(cv_accs)
    best_roc_idx = np.argmax(roc_aucs)
    best_keep_loss_idx = np.argmin([k for k in keep_loss_rates if k > 0]) if any(k > 0 for k in keep_loss_rates) else 0
    
    print(f"\nBest CV Accuracy: {cv_accs[best_cv_idx]:.4f}±{cv_stds[best_cv_idx]:.4f} at {param_name}={param_values[best_cv_idx]}")
    print(f"Best ROC-AUC: {roc_aucs[best_roc_idx]:.4f} at {param_name}={param_values[best_roc_idx]}")
    if keep_loss_rates[best_keep_loss_idx] > 0:
        print(f"Best Keep-Loss Rate: {keep_loss_rates[best_keep_loss_idx]:.2%} at {param_name}={param_values[best_keep_loss_idx]}")
    print(f"Training time at best CV: {training_times[best_cv_idx]:.1f}s")

def main():
    # Use default dataset path
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        return
    
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    if not os.path.exists(repo_path):
        logging.error(f"Repository file not found: {repo_path}")
        return
    
    repo = RatingsTagsRepository(repo_path)
    
    print("=" * 80)
    print("FAST MODE HYPERPARAMETER STUDY WITH CV (ROBUST EVALUATION)")
    print("=" * 80)
    print("\nThis will test each hyperparameter independently with CV enabled.")
    print("Base settings: LR=0.05, iterations=1000, patience=100")
    print("Ranges: LR=0.0001-0.5 (log scale), iterations=10-2000, patience=1-200")
    print("~50 points per parameter, using 5-fold CV for robust evaluation")
    print("\n⚠️  WARNING: This will take much longer (~50x per parameter)")
    print()
    
    # Test 1: Learning Rate (0.0001 to 0.5, log scale)
    print("\n" + "=" * 80)
    print("STUDY 1: Learning Rate Impact (0.0001 to 0.5) - WITH CV")
    print("=" * 80)
    # Use fewer points for CV version (takes longer)
    lr_values = np.logspace(np.log10(0.0001), np.log10(0.5), 20)  # Reduced to 20 for speed
    lr_results = test_hyperparameter(
        image_dir, repo, "learning_rate", lr_values,
        base_lr=0.05, base_iterations=1000, base_patience=100
    )
    plot_results("learning_rate", lr_results)
    
    # Test 2: Max Iterations (10 to 2000, linear scale)
    print("\n" + "=" * 80)
    print("STUDY 2: Max Iterations Impact (10 to 2000) - WITH CV")
    print("=" * 80)
    # Use fewer points for CV version
    iterations_values = np.linspace(10, 2000, 20, dtype=int)  # Reduced to 20 for speed
    iterations_results = test_hyperparameter(
        image_dir, repo, "max_iterations", iterations_values,
        base_lr=0.05, base_iterations=1000, base_patience=100
    )
    plot_results("max_iterations", iterations_results)
    
    # Test 3: Patience (1 to 200, linear scale)
    print("\n" + "=" * 80)
    print("STUDY 3: Patience Impact (1 to 200) - WITH CV")
    print("=" * 80)
    # Use fewer points for CV version
    patience_values = np.linspace(1, 200, 20, dtype=int)  # Reduced to 20 for speed
    patience_results = test_hyperparameter(
        image_dir, repo, "patience", patience_values,
        base_lr=0.05, base_iterations=1000, base_patience=100
    )
    plot_results("patience", patience_results)
    
    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print("\nPlots saved in plots/ directory:")
    print("  - plots/fast_mode_learning_rate_study_cv.png")
    print("  - plots/fast_mode_max_iterations_study_cv.png")
    print("  - plots/fast_mode_patience_study_cv.png")
    print("\nThese plots show CV metrics (more robust than single test set)")

if __name__ == "__main__":
    main()

