#!/usr/bin/env python3
"""Train model on combined dataset (current + AVA) to prevent overfitting.

This script:
1. Loads current dataset
2. Optionally loads AVA dataset labels
3. Trains model with proper validation
4. Detects and reports overfitting
5. Evaluates on held-out test set
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.model_version import create_model_metadata
from src.training_core import DEFAULT_MODEL_PATH
from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("train_ava")
log.setLevel(logging.INFO)


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def align_and_concat(X_feats: np.ndarray, filenames: list[str], emb: np.ndarray, emb_fnames: list[str]):
    """Align embeddings with features by filename and concatenate."""
    emb_map = {os.path.basename(f): i for i, f in enumerate(emb_fnames)}
    rows = []
    for fn in filenames:
        basename = os.path.basename(fn)
        if basename in emb_map:
            rows.append(emb[emb_map[basename]])
        else:
            rows.append(np.zeros((emb.shape[1],), dtype=float))
    emb_mat = np.vstack(rows)
    return np.hstack([X_feats, emb_mat])


def add_interactions(X: np.ndarray, top_n: int = 15, max_interactions: int = 100):
    """Add feature interactions (memory-efficient)."""
    n_select = min(top_n, X.shape[1])
    feature_vars = np.var(X, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:n_select]
    X_selected = X[:, top_indices]
    
    interactions = []
    interaction_pairs = []
    for i in range(n_select):
        for j in range(i + 1, n_select):
            interaction = X_selected[:, i] * X_selected[:, j]
            interactions.append(interaction)
            interaction_pairs.append((int(top_indices[i]), int(top_indices[j])))
            if len(interactions) >= max_interactions:
                break
        if len(interactions) >= max_interactions:
            break
    
    if interactions:
        X_interactions = np.column_stack(interactions)
        return np.hstack([X, X_interactions]), interaction_pairs
    return X, []


def add_ratio_features(X: np.ndarray, top_indices: list[int] = None, n_ratios: int = 20):
    """Add ratio features from top important features."""
    if top_indices is None:
        top_indices = list(range(min(10, X.shape[1])))
    
    ratios = []
    ratio_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            idx_i = top_indices[i]
            idx_j = top_indices[j]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(X[:, idx_i], X[:, idx_j], 
                                out=np.zeros_like(X[:, idx_i]), 
                                where=X[:, idx_j]!=0)
                ratios.append(ratio)
                ratio_pairs.append((idx_i, idx_j))
            if len(ratios) >= n_ratios:
                break
        if len(ratios) >= n_ratios:
            break
    
    if ratios:
        X_ratios = np.column_stack(ratios)
        return np.hstack([X, X_ratios]), ratio_pairs
    return X, []


def main():
    parser = argparse.ArgumentParser(description="Train model on combined dataset")
    parser.add_argument("--image-dir", default=None, help="Current image directory")
    parser.add_argument("--ava-labels", default=None, help="AVA labels JSON file")
    parser.add_argument("--ava-images-dir", default=None, help="Directory with AVA images (if downloaded)")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings file")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension for embeddings")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation set size")
    parser.add_argument("--max-ava", type=int, default=None, help="Maximum AVA samples to use")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    log.info("="*80)
    log.info("TRAINING ON COMBINED DATASET (Current + AVA)")
    log.info("="*80)
    
    # Load current dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if not repo:
        log.error("No repository found")
        return 1
    
    # Load AVA features only (no current dataset)
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    X_ava = None
    y_ava = None
    
    ava_features_path = cache_dir / "ava_features.joblib"
    if not ava_features_path.exists():
        log.error(f"AVA features not found at {ava_features_path}")
        log.error("Run: poetry run python scripts/extract_ava_features.py --ava-images-dir .cache/ava_dataset/images --ava-labels .cache/ava_dataset/ava_keep_trash_labels.json")
        return 1
    
    log.info(f"Loading AVA features from {ava_features_path}...")
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    y_ava = ava_data['labels']
    
    if args.max_ava and len(y_ava) > args.max_ava:
        log.info(f"Limiting AVA to {args.max_ava} samples")
        X_ava = X_ava[:args.max_ava]
        y_ava = y_ava[:args.max_ava]
    
    log.info(f"AVA dataset: {len(y_ava)} samples")
    log.info(f"  Keep: {np.sum(y_ava == 1)} ({np.sum(y_ava == 1)/len(y_ava)*100:.1f}%)")
    log.info(f"  Trash: {np.sum(y_ava == 0)} ({np.sum(y_ava == 0)/len(y_ava)*100:.1f}%)")
    
    # Use AVA dataset only
    X = X_ava
    y = y_ava
    filenames = [f"ava_{i}" for i in range(len(y_ava))]
    
    log.info(f"\nTotal dataset: {len(y)} samples")
    
    # Load embeddings
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    embeddings_path = args.embeddings
    if not embeddings_path:
        possible = [
            cache_dir / "embeddings_resnet18_full.joblib",
            cache_dir / "embeddings_resnet18.joblib",
        ]
        for p in possible:
            if p.exists():
                embeddings_path = str(p)
                break
    
    if not embeddings_path or not os.path.exists(embeddings_path):
        log.error("No embeddings file found")
        return 1
    
    log.info(f"Loading embeddings from {embeddings_path}...")
    emb, emb_fnames = load_embeddings(embeddings_path)
    
    # Apply PCA if requested
    if args.pca_dim and args.pca_dim < emb.shape[1]:
        log.info(f"Applying PCA to reduce embeddings to {args.pca_dim} dimensions...")
        pca = PCA(n_components=args.pca_dim, random_state=42)
        emb = pca.fit_transform(emb)
        log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        pca = None
    
    # Concatenate features and embeddings
    X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    log.info(f"Base features: {X_combined.shape[1]} total ({X.shape[1]} handcrafted + {emb.shape[1]} embeddings)")
    
    # Add feature interactions
    X_final, interaction_pairs = add_interactions(X_combined, top_n=15, max_interactions=100)
    X_final, ratio_pairs = add_ratio_features(X_final, top_indices=list(range(min(10, X.shape[1]))), n_ratios=20)
    
    log.info(f"Final feature count: {X_final.shape[1]} (base: {X_combined.shape[1]}, interactions: {len(interaction_pairs)}, ratios: {len(ratio_pairs)})")
    
    # Split: train / validation / test
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y, test_size=args.test_size, stratify=y, random_state=42
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_size/(1-args.test_size), stratify=y_trainval, random_state=42
    )
    
    log.info(f"\nData splits:")
    log.info(f"  Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
    log.info(f"  Validation: {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)")
    log.info(f"  Test: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
    
    # Load tuned CatBoost hyperparameters
    tuning_path = cache_dir / "catboost_tuning_results.json"
    
    if tuning_path.exists():
        log.info(f"Loading tuned hyperparameters from {tuning_path}...")
        with open(tuning_path) as f:
            tuning_data = json.load(f)
            cb_params = tuning_data["best_params"]
        log.info(f"  Iterations: {cb_params.get('iterations', 'default')}")
        log.info(f"  Learning rate: {cb_params.get('learning_rate', 'default')}")
        log.info(f"  Depth: {cb_params.get('depth', 'default')}")
        log.info(f"  L2 regularization: {cb_params.get('l2_leaf_reg', 'default')}")
    else:
        log.info("Using default hyperparameters (no tuning file found)")
        cb_params = {
            "iterations": 500,
            "learning_rate": 0.07,
            "depth": 9,
            "l2_leaf_reg": 2.48,
            "border_count": 81,
        }
    
    # Optimize for validation accuracy with strong regularization and dropout
    log.info("\nOptimizing hyperparameters for validation accuracy...")
    log.info("Configuration: iterations=2000, depth=5, l2=10.0, lr=0.1, dropout=0.3, early_stopping=50")
    
    # Strategy: Strong regularization + dropout + higher learning rate
    cb_params["iterations"] = 2000  # Much bigger max iterations
    cb_params["l2_leaf_reg"] = 10.0  # Strong L2 regularization
    cb_params["depth"] = 5  # Shallow trees (stronger regularization)
    cb_params["learning_rate"] = 0.1  # Higher learning rate
    cb_params["rsm"] = 0.7  # Random Subspace Method (dropout-like): use 70% of features per split
    cb_params["subsample"] = 0.8  # Row subsampling: use 80% of samples per tree
    cb_params["bagging_temperature"] = 1.0  # Bayesian bagging (adds randomness)
    cb_params["random_strength"] = 1.0  # Random strength for splits
    
    log.info(f"  Adjusted iterations: {cb_params['iterations']}")
    log.info(f"  Adjusted learning rate: {cb_params['learning_rate']}")
    log.info(f"  Adjusted depth: {cb_params['depth']}")
    log.info(f"  Adjusted L2 regularization: {cb_params['l2_leaf_reg']}")
    log.info(f"  RSM (feature dropout): {cb_params.get('rsm', 'default')}")
    log.info(f"  Subsample (row dropout): {cb_params.get('subsample', 'default')}")
    
    # Train CatBoost
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return 1
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"\nClass distribution:")
    log.info(f"  Keep: {n_keep} ({n_keep/len(y_train)*100:.1f}%)")
    log.info(f"  Trash: {n_trash} ({n_trash/len(y_train)*100:.1f}%)")
    log.info(f"  Scale pos weight: {scale_pos_weight:.3f}")
    
    cb_params_final = {
        **cb_params,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": 100,  # Show progress every 100 iterations
        "thread_count": -1,
        "use_best_model": False,  # Use final model (not best from validation)
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params_final)),
    ])
    
    log.info("\n" + "="*80)
    log.info("TRAINING CATBOOST MODEL")
    log.info("="*80)
    log.info(f"Training set size: {len(y_train)}")
    log.info(f"Validation set size: {len(y_val)}")
    log.info(f"Feature count: {X_train.shape[1]}")
    log.info(f"Starting training with early stopping...")
    log.info("")
    
    # Fit with validation set for monitoring and manual early stopping
    log.info("\n" + "="*80)
    log.info("TRAINING CONFIGURATION")
    log.info("="*80)
    log.info(f"MAX ITERATIONS: {cb_params_final['iterations']}")
    log.info(f"EARLY STOPPING: Will stop if validation accuracy doesn't improve for 200 iterations")
    log.info(f"CHECK INTERVAL: Every 50 iterations")
    log.info(f"STOPPING CONDITIONS:")
    log.info(f"  1. Early stopping: No improvement for 200 iterations")
    log.info(f"  2. Max iterations: Reaches {cb_params_final['iterations']} iterations")
    log.info("="*80)
    log.info("")
    
    # Manual early stopping: train in chunks and check validation accuracy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_val_acc = 0.0
    best_iter = 0
    no_improve_count = 0
    check_interval = 50
    early_stopping_rounds = 200
    
    # Start with initial model
    cat_model = None
    
    for chunk_start in range(0, cb_params_final['iterations'], check_interval):
        chunk_end = min(chunk_start + check_interval, cb_params_final['iterations'])
        
        log.info("\n" + "="*80)
        log.info(f"TRAINING: Iterations {chunk_start} to {chunk_end} (of {cb_params_final['iterations']} max)")
        log.info("="*80)
        
        if cat_model is None:
            # First chunk: create new model
            log.info("Creating new model and training...")
            cat_model = CatBoostClassifier(**{**cb_params_final, "iterations": chunk_end})
            cat_model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                verbose=50,  # Show progress every 50 iterations
            )
        else:
            # Continue training: create new model with more iterations
            log.info("Continuing training with more iterations...")
            cat_model_new = CatBoostClassifier(**{**cb_params_final, "iterations": chunk_end})
            cat_model_new.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                verbose=50,
            )
            cat_model = cat_model_new
        
        log.info("Evaluating model performance...")
        # Check training and validation accuracy
        y_train_proba = cat_model.predict_proba(X_train_scaled)[:, 1]
        y_train_pred = (y_train_proba >= 0.67).astype(int)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        y_val_proba = cat_model.predict_proba(X_val_scaled)[:, 1]
        y_val_pred = (y_val_proba >= 0.67).astype(int)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        log.info("")
        log.info("="*80)
        log.info(f"RESULTS AT ITERATION {chunk_end}")
        log.info("="*80)
        log.info(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        log.info(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        log.info("="*80)
        
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_iter = chunk_end
            no_improve_count = 0
            log.info("")
            log.info("✓✓✓ VALIDATION ACCURACY IMPROVED! ✓✓✓")
            log.info(f"   New best: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at iteration {best_iter}")
            log.info(f"   Improvement: +{improvement*100:.2f} percentage points")
            log.info("")
        else:
            no_improve_count += check_interval
            log.info("")
            log.info(f"⚠ No improvement in validation accuracy")
            log.info(f"   Current: {val_acc:.4f} ({val_acc*100:.2f}%)")
            log.info(f"   Best:    {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at iteration {best_iter}")
            log.info(f"   No improvement for: {no_improve_count} iterations (patience: {early_stopping_rounds})")
            log.info("")
            
            if no_improve_count >= early_stopping_rounds:
                log.info("")
                log.info("="*80)
                log.info("🛑 EARLY STOPPING TRIGGERED 🛑")
                log.info("="*80)
                log.info(f"Validation accuracy hasn't improved for {no_improve_count} iterations")
                log.info(f"Best model found at iteration {best_iter} with validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
                log.info(f"Training will stop and use the best model.")
                log.info("="*80)
                log.info("")
                # Use best model
                log.info("Retraining best model...")
                cat_model = CatBoostClassifier(**{**cb_params_final, "iterations": best_iter})
                cat_model.fit(X_train_scaled, y_train)
                break
        
        # Check if reached max iterations
        if chunk_end >= cb_params_final['iterations']:
            log.info("")
            log.info("="*80)
            log.info("✓ MAX ITERATIONS REACHED ✓")
            log.info("="*80)
            log.info(f"Reached maximum iterations: {cb_params_final['iterations']}")
            log.info(f"Best model found at iteration {best_iter} with validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
            log.info("="*80)
            log.info("")
    
    clf = Pipeline([
        ("scaler", scaler),
        ("cat", cat_model),
    ])
    
    log.info(f"\nTraining completed: Best iteration was {best_iter} (validation accuracy: {best_val_acc:.4f})")
    
    log.info("\n" + "="*80)
    log.info("EVALUATING MODEL")
    log.info("="*80)
    
    # Evaluate on validation set (for overfitting detection)
    log.info("\nValidation set evaluation...")
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba >= 0.67).astype(int)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    log.info(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    log.info(f"  Precision: {val_precision:.4f}")
    log.info(f"  Recall: {val_recall:.4f}")
    log.info(f"  F1: {val_f1:.4f}")
    
    # Evaluate on training set
    log.info("\nTraining set evaluation...")
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_proba >= 0.67).astype(int)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    log.info(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    log.info(f"  Precision: {train_precision:.4f}")
    log.info(f"  Recall: {train_recall:.4f}")
    log.info(f"  F1: {train_f1:.4f}")
    
    # Check for overfitting
    overfitting_gap = train_accuracy - val_accuracy
    log.info("\n" + "="*80)
    log.info("OVERFITTING DETECTION")
    log.info("="*80)
    log.info(f"Training accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    log.info(f"Validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    log.info(f"Gap:                  {overfitting_gap:.4f} ({overfitting_gap*100:.2f} percentage points)")
    
    if overfitting_gap > 0.10:
        log.warning("⚠️  Significant overfitting detected (gap > 10%)")
        log.warning("   Consider: more data, regularization, or early stopping")
    elif overfitting_gap > 0.05:
        log.warning("⚠️  Moderate overfitting detected (gap > 5%)")
    else:
        log.info("✓ No significant overfitting detected")
    
    # Evaluate on test set
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.67).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    log.info("\n" + "="*80)
    log.info("TEST SET EVALUATION")
    log.info("="*80)
    log.info(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    log.info(f"Precision: {test_precision:.4f}")
    log.info(f"Recall:    {test_recall:.4f}")
    log.info(f"F1:        {test_f1:.4f}")
    log.info(f"ROC-AUC:   {test_roc_auc:.4f}" if not np.isnan(test_roc_auc) else "ROC-AUC:   N/A")
    log.info(f"\nConfusion Matrix:")
    log.info(f"  True Negatives:  {tn}")
    log.info(f"  False Positives: {fp}")
    log.info(f"  False Negatives: {fn}")
    log.info(f"  True Positives:  {tp}")
    
    # Create feature interaction transformer
    feature_transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    
    # Save model
    metadata = create_model_metadata(
        feature_count=X.shape[1],
        feature_mode="FULL",
        params=cb_params_final,
        n_samples=len(y_train),
    )
    metadata.update({
        "model_type": "CatBoost",
        "has_embeddings": True,
        "n_embedding_features": int(emb.shape[1]),
        "has_feature_interactions": True,
        "n_interaction_features": len(interaction_pairs),
        "n_ratio_features": len(ratio_pairs),
        "n_base_features": int(X_combined.shape[1]),
        "n_total_features": int(X_final.shape[1]),
        "optimal_threshold": 0.67,
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "overfitting_gap": float(overfitting_gap),
    })
    
    model_data = {
        "__metadata__": metadata,
        "model": clf,
        "pca": pca,
        "feature_transformer": feature_transformer,
        "feature_length": int(X_final.shape[1]),
        "n_base_features": int(X_combined.shape[1]),
        "optimal_threshold": 0.67,
    }
    
    joblib.dump(model_data, DEFAULT_MODEL_PATH)
    log.info(f"\nModel saved to: {DEFAULT_MODEL_PATH}")
    
    # Save results
    results = {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "overfitting_gap": float(overfitting_gap),
        "test_metrics": {
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1": float(test_f1),
            "roc_auc": float(test_roc_auc) if not np.isnan(test_roc_auc) else None,
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    
    results_path = cache_dir / "train_with_ava_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

