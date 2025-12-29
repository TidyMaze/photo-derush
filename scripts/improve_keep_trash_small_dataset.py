#!/usr/bin/env python3
"""Improve keep/trash model for SMALL datasets (adapted from AVA learnings).

Key differences from AVA:
- Less aggressive regularization (small datasets need less)
- Fewer interactions (avoid overfitting)
- Simpler params that work on small data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.feature_transformer import FeatureInteractionTransformer
from src.repository import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("improve_small")


def load_embeddings(path: str):
    data = joblib.load(path)
    return data['embeddings'], data.get('filenames', [])


def align_and_concat(X_feats: np.ndarray, filenames: list[str], emb: np.ndarray, emb_fnames: list[str]):
    import os
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


def prepare_features_with_interactions(X_handcrafted, embeddings_path=None, pca_dim=128, n_interactions=50):
    """Prepare features with LIMITED interactions for small datasets."""
    X = X_handcrafted
    filenames = [f"img_{i}" for i in range(len(X_handcrafted))]
    
    if embeddings_path and os.path.exists(embeddings_path):
        log.info(f"Loading embeddings from {embeddings_path}")
        emb, emb_fnames = load_embeddings(embeddings_path)
        
        if pca_dim and pca_dim < emb.shape[1]:
            log.info(f"Applying PCA: {emb.shape[1]} → {pca_dim} dimensions")
            pca = PCA(n_components=pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
        
        X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    else:
        log.warning("No embeddings found, using handcrafted features only")
        X_combined = X
    
    log.info(f"Base features: {X_combined.shape[1]}")
    
    # LIMITED interactions for small datasets
    log.info(f"Adding LIMITED feature interactions ({n_interactions} interactions)...")
    top_n = 10  # Fewer top features for small datasets
    feature_vars = np.var(X_combined, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:top_n]
    
    interaction_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < n_interactions:
                interaction_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    ratio_pairs = []
    for i in range(min(10, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 3, len(top_indices))):
            if len(ratio_pairs) < 10:  # Fewer ratios
                ratio_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    log.info(f"Final features: {X_final.shape[1]} (with {len(interaction_pairs)} interactions + {len(ratio_pairs)} ratios)")
    
    return X_final, transformer


def train_improved_model(X_train, X_val, X_test, y_train, y_val, y_test, n_samples):
    """Train model with SMALL-DATASET-ADAPTED params."""
    from catboost import CatBoostClassifier
    
    # Adapt params based on dataset size
    if n_samples < 500:
        # Small dataset: less regularization, fewer iterations
        cb_params = {
            "iterations": 500,  # Fewer iterations
            "learning_rate": 0.05,  # Higher learning rate
            "depth": 6,  # Shallower trees
            "l2_leaf_reg": 1.0,  # Less L2 regularization
            "rsm": 0.95,  # Less feature dropout
            "bootstrap_type": "Bernoulli",
            "subsample": 0.95,  # Less row dropout
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "loss_function": "Logloss",
            "early_stopping_rounds": 50,  # Earlier stopping
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }
        log.info("Using SMALL-DATASET-ADAPTED hyperparameters")
    else:
        # Larger dataset: use AVA params
        cb_params = {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "rsm": 0.88,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "loss_function": "Logloss",
            "early_stopping_rounds": 200,
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }
        log.info("Using AVA-learned hyperparameters")
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    cb_params["scale_pos_weight"] = scale_pos_weight
    
    log.info(f"Class balance: {n_keep} keep, {n_trash} trash (scale_pos_weight={scale_pos_weight:.3f})")
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    log.info("Training model...")
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    cat_model = clf.named_steps['cat']
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_val_pred = cat_model.predict(X_val_scaled)
    y_test_pred = cat_model.predict(X_test_scaled)
    y_test_proba = cat_model.predict_proba(X_test_scaled)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    log.info(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    log.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    log.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    log.info(f"\nConfusion Matrix:")
    log.info(f"  TN={tn}, FP={fp}")
    log.info(f"  FN={fn}, TP={tp}")
    
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_test_pred, target_names=["trash", "keep"]))
    
    return clf, {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "roc_auc": test_roc_auc,
        "confusion_matrix": (tn, fp, fn, tp),
    }


def main():
    parser = argparse.ArgumentParser(description="Improve keep/trash model (small dataset adapted)")
    parser.add_argument("--image-dir", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings file")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension")
    parser.add_argument("--n-interactions", type=int, default=50, help="Number of interactions")
    parser.add_argument("--model-path", default=None, help="Path to save model")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.exists(image_dir):
        log.error(f"Image directory not found: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("IMPROVING KEEP/TRASH MODEL (SMALL DATASET ADAPTED)")
    log.info("="*80)
    log.info(f"Image directory: {image_dir}")
    
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if not repo:
        log.error("No repository found")
        return 1
    
    log.info("Building dataset...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(y)
    log.info(f"Dataset: {n_samples} samples ({np.sum(y==1)} keep, {np.sum(y==0)} trash)")
    
    if n_samples < 20:
        log.error("Insufficient labeled data")
        return 1
    
    # Find embeddings
    if args.embeddings:
        embeddings_path = args.embeddings
    else:
        for path in [".cache/embeddings_resnet18.joblib", ".cache/embeddings_resnet18_full.joblib"]:
            if os.path.exists(path):
                embeddings_path = path
                break
        else:
            embeddings_path = None
    
    # Prepare features with LIMITED interactions
    X_final, transformer = prepare_features_with_interactions(
        X, embeddings_path, args.pca_dim, args.n_interactions
    )
    
    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
    )
    
    log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    model, metrics = train_improved_model(
        X_train, X_val, X_test, y_train, y_val, y_test, n_samples
    )
    
    baseline_acc = 0.8667
    improvement = metrics["test_acc"] - baseline_acc
    log.info(f"\nBaseline (current model): {baseline_acc*100:.2f}%")
    log.info(f"Improved model: {metrics['test_acc']*100:.2f}%")
    log.info(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0.01:
        log.info("\n✅ Model improved with adapted AVA learnings!")
    elif improvement > -0.05:
        log.info("\n⚠️  Model similar to baseline (small dataset limitations)")
    else:
        log.info("\n❌ Model performed worse (interactions may be overfitting)")
        log.info("   Recommendation: Use current model or try without interactions")
    
    # Save
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.expanduser("~/.photo-derush-improved-small.joblib")
    
    model_data = {
        "model": model,
        "feature_transformer": transformer,
        "metrics": metrics,
        "pca_dim": args.pca_dim,
        "n_interactions": args.n_interactions,
    }
    
    joblib.dump(model_data, model_path)
    log.info(f"\nModel saved to: {model_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

