#!/usr/bin/env python3
"""Test if doubling max_iterations/patience and halving learning_rate improves fast mode."""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training_core import train_keep_trash_model, DEFAULT_MODEL_PATH
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    print("FAST MODE HYPERPARAMETER TEST")
    print("=" * 80)
    print()
    
    # Test 1: Current fast mode settings
    print("TEST 1: Current Fast Mode Settings")
    print("-" * 80)
    print("Learning rate: 0.1")
    print("Max iterations: 500")
    print("Patience: 50")
    print()
    
    t1_start = time.perf_counter()
    result1 = train_keep_trash_model(
        image_dir=image_dir,
        repo=repo,
        model_path=DEFAULT_MODEL_PATH.replace(".joblib", "_fast_current.joblib"),
        fast_mode=True,
        early_stopping_rounds=50,
    )
    t1_time = time.perf_counter() - t1_start
    
    if result1:
        print(f"✓ Current fast mode:")
        if result1.cv_accuracy_mean:
            print(f"  - CV Accuracy: {result1.cv_accuracy_mean:.2%} ± {result1.cv_accuracy_std:.2%}" if result1.cv_accuracy_std else f"  - CV Accuracy: {result1.cv_accuracy_mean:.2%}")
        if result1.f1:
            print(f"  - F1 Score: {result1.f1:.4f}")
        if result1.roc_auc:
            print(f"  - ROC-AUC: {result1.roc_auc:.4f}")
        if result1.aggregated_metrics:
            keep_loss = result1.aggregated_metrics.get('keep_loss_rate_mean')
            if keep_loss:
                print(f"  - Keep-Loss Rate: {keep_loss:.2%}")
        print(f"  - Samples: {result1.n_samples} ({result1.n_keep} keep, {result1.n_trash} trash)")
        print(f"  - Training time: {t1_time:.2f}s")
        print()
    else:
        print("✗ Current fast mode training failed")
        return
    
    # Test 2: Improved fast mode settings (double iterations/patience, half LR)
    print("TEST 2: Improved Fast Mode Settings")
    print("-" * 80)
    print("Learning rate: 0.05 (halved)")
    print("Max iterations: 1000 (doubled)")
    print("Patience: 100 (doubled)")
    print()
    
    # Temporarily modify _build_pipeline to use new settings
    from src import training_core
    original_build_pipeline = training_core._build_pipeline
    
    def modified_build_pipeline(
        random_state: int, scale_pos_weight: float, xgb_params: dict, 
        early_stopping_rounds: int | None, use_catboost: bool = True, 
        n_handcrafted_features: int = 78, fast_mode: bool = False
    ):
        """Modified _build_pipeline with improved fast mode settings."""
        if use_catboost and fast_mode:
            try:
                from catboost import CatBoostClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                import os
                
                # Improved fast mode settings
                max_iterations = 1000  # Doubled from 500
                effective_patience = max(early_stopping_rounds, 100) if early_stopping_rounds else None  # Doubled from 50
                learning_rate = 0.05  # Halved from 0.1
                
                cb_params = {
                    "iterations": max_iterations,
                    "learning_rate": learning_rate,
                    "depth": 6,
                    "l2_leaf_reg": 1.0,
                    "scale_pos_weight": scale_pos_weight,
                    "random_seed": random_state,
                    "verbose": False,
                    "thread_count": 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1,
                }
                
                if early_stopping_rounds is not None:
                    cb_params["early_stopping_rounds"] = effective_patience
                    cb_params["eval_metric"] = "Logloss"
                    cb_params["use_best_model"] = True
                    cb_params["verbose"] = 10
                    logging.info(f"[train] Improved fast mode: patience={effective_patience}, max_iterations={max_iterations}, LR={learning_rate}, eval_metric=Logloss")
                else:
                    logging.info(f"[train] Improved fast mode: {max_iterations} iterations, LR={learning_rate}")
                
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("cat", CatBoostClassifier(**cb_params)),
                ])
            except ImportError:
                pass
        
        # Fallback to original for non-fast-mode or if CatBoost unavailable
        return original_build_pipeline(
            random_state, scale_pos_weight, xgb_params, 
            early_stopping_rounds, use_catboost, n_handcrafted_features, fast_mode
        )
    
    # Monkey patch for this test
    training_core._build_pipeline = modified_build_pipeline
    
    try:
        t2_start = time.perf_counter()
        result2 = train_keep_trash_model(
            image_dir=image_dir,
            repo=repo,
            model_path=DEFAULT_MODEL_PATH.replace(".joblib", "_fast_improved.joblib"),
            fast_mode=True,
            early_stopping_rounds=100,  # Will be used as min(100, 100) = 100
        )
        t2_time = time.perf_counter() - t2_start
        
        if result2:
            print(f"✓ Improved fast mode:")
            if result2.cv_accuracy_mean:
                print(f"  - CV Accuracy: {result2.cv_accuracy_mean:.2%} ± {result2.cv_accuracy_std:.2%}" if result2.cv_accuracy_std else f"  - CV Accuracy: {result2.cv_accuracy_mean:.2%}")
            if result2.f1:
                print(f"  - F1 Score: {result2.f1:.4f}")
            if result2.roc_auc:
                print(f"  - ROC-AUC: {result2.roc_auc:.4f}")
            if result2.aggregated_metrics:
                keep_loss = result2.aggregated_metrics.get('keep_loss_rate_mean')
                if keep_loss:
                    print(f"  - Keep-Loss Rate: {keep_loss:.2%}")
            print(f"  - Samples: {result2.n_samples} ({result2.n_keep} keep, {result2.n_trash} trash)")
            print(f"  - Training time: {t2_time:.2f}s")
            print()
        else:
            print("✗ Improved fast mode training failed")
            return
        
        # Compare results
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print()
        
        # Compare CV accuracy
        if result1.cv_accuracy_mean and result2.cv_accuracy_mean:
            accuracy_diff = result2.cv_accuracy_mean - result1.cv_accuracy_mean
            acc1_str = f"{result1.cv_accuracy_mean:.2%} ± {result1.cv_accuracy_std:.2%}" if result1.cv_accuracy_std else f"{result1.cv_accuracy_mean:.2%}"
            acc2_str = f"{result2.cv_accuracy_mean:.2%} ± {result2.cv_accuracy_std:.2%}" if result2.cv_accuracy_std else f"{result2.cv_accuracy_mean:.2%}"
            print(f"CV Accuracy:  {acc1_str} → {acc2_str} ({accuracy_diff:+.2%})")
        
        # Compare F1
        if result1.f1 and result2.f1:
            f1_diff = result2.f1 - result1.f1
            print(f"F1 Score:     {result1.f1:.4f} → {result2.f1:.4f} ({f1_diff:+.4f})")
        
        # Compare ROC-AUC
        if result1.roc_auc and result2.roc_auc:
            roc_diff = result2.roc_auc - result1.roc_auc
            print(f"ROC-AUC:     {result1.roc_auc:.4f} → {result2.roc_auc:.4f} ({roc_diff:+.4f})")
        
        # Compare Keep-Loss Rate
        if result1.aggregated_metrics and result2.aggregated_metrics:
            keep_loss1 = result1.aggregated_metrics.get('keep_loss_rate_mean')
            keep_loss2 = result2.aggregated_metrics.get('keep_loss_rate_mean')
            if keep_loss1 and keep_loss2:
                keep_loss_diff = keep_loss2 - keep_loss1
                print(f"Keep-Loss:    {keep_loss1:.2%} → {keep_loss2:.2%} ({keep_loss_diff:+.2%})")
        
        # Compare training time
        time_diff = t2_time - t1_time
        print(f"Training Time: {t1_time:.2f}s → {t2_time:.2f}s ({time_diff:+.2f}s)")
        print()
        
        # Overall assessment
        if result1.cv_accuracy_mean and result2.cv_accuracy_mean:
            accuracy_diff = result2.cv_accuracy_mean - result1.cv_accuracy_mean
            if accuracy_diff > 0.01:  # >1% improvement
                print("✅ IMPROVED SETTINGS ARE BETTER - CV Accuracy improved by >1%")
            elif accuracy_diff < -0.01:  # >1% worse
                print("❌ IMPROVED SETTINGS ARE WORSE - CV Accuracy decreased by >1%")
            else:
                print("➡️  IMPROVED SETTINGS ARE SIMILAR - CV Accuracy change <1%")
        
        if result1.aggregated_metrics and result2.aggregated_metrics:
            keep_loss1 = result1.aggregated_metrics.get('keep_loss_rate_mean')
            keep_loss2 = result2.aggregated_metrics.get('keep_loss_rate_mean')
            if keep_loss1 and keep_loss2:
                keep_loss_diff = keep_loss2 - keep_loss1
                if keep_loss_diff < -0.005:  # >0.5% improvement (lower is better)
                    print("✅ Keep-Loss Rate improved (lower is better)")
                elif keep_loss_diff > 0.005:  # >0.5% worse
                    print("❌ Keep-Loss Rate worsened (higher is worse)")
                else:
                    print("➡️  Keep-Loss Rate similar")
        
        # Time assessment
        if time_diff > 2.0:  # >2s slower
            print(f"⚠️  WARNING: Training time increased by {time_diff:.2f}s (may exceed 5s target)")
        elif time_diff < -0.5:  # >0.5s faster
            print(f"✅ Training time improved by {abs(time_diff):.2f}s")
        else:
            print("➡️  Training time similar")
        
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        if result1.roc_auc and result2.roc_auc:
            roc_improvement = result2.roc_auc - result1.roc_auc
            if roc_improvement > 0.05:  # >5% improvement
                print("✅ STRONG IMPROVEMENT: ROC-AUC improved significantly (+{:.1%})".format(roc_improvement))
            elif roc_improvement > 0.01:  # >1% improvement
                print("✅ MODERATE IMPROVEMENT: ROC-AUC improved (+{:.1%})".format(roc_improvement))
            else:
                print("➡️  MINIMAL CHANGE: ROC-AUC change <1%")
        
        if time_diff <= 2.0 and t2_time <= 5.0:
            print("✅ Training time acceptable (<5s target)")
        elif time_diff > 2.0:
            print("⚠️  Training time increased significantly")
        
    finally:
        # Restore original function
        training_core._build_pipeline = original_build_pipeline

if __name__ == "__main__":
    main()

