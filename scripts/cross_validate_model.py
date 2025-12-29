#!/usr/bin/env python3
"""Cross-validate model to better estimate true performance and detect overfitting.

Uses k-fold cross-validation to get more reliable performance estimates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("cross_validate")


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    import joblib
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
    """Add feature interactions."""
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
    """Add ratio features."""
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
    parser = argparse.ArgumentParser(description="Cross-validate model")
    parser.add_argument("--image-dir", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings file")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    log.info("="*80)
    log.info("CROSS-VALIDATION EVALUATION")
    log.info("="*80)
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if not repo:
        log.error("No repository found")
        return 1
    
    log.info(f"Loading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples")
    
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
    
    # Apply PCA
    if args.pca_dim and args.pca_dim < emb.shape[1]:
        log.info(f"Applying PCA to {args.pca_dim} dimensions...")
        pca = PCA(n_components=args.pca_dim, random_state=42)
        emb = pca.fit_transform(emb)
    else:
        pca = None
    
    # Concatenate features
    X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    X_final, interaction_pairs = add_interactions(X_combined, top_n=15, max_interactions=100)
    X_final, ratio_pairs = add_ratio_features(X_final, top_indices=list(range(min(10, X.shape[1]))), n_ratios=20)
    
    log.info(f"Features: {X_final.shape[1]} total")
    
    # Cross-validation
    from catboost import CatBoostClassifier
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_roc_aucs = []
    
    log.info(f"\nRunning {args.n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y), 1):
        log.info(f"\nFold {fold}/{args.n_folds}...")
        
        X_train_fold = X_final[train_idx]
        X_val_fold = X_final[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Train
        n_keep = int(np.sum(y_train_fold == 1))
        n_trash = int(np.sum(y_train_fold == 0))
        scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
        
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("cat", CatBoostClassifier(
                iterations=500,
                learning_rate=0.07,
                depth=9,
                l2_leaf_reg=2.48,
                border_count=81,
                scale_pos_weight=scale_pos_weight,
                random_seed=42,
                verbose=False,
                thread_count=-1,
            )),
        ])
        
        clf.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_val_proba = clf.predict_proba(X_val_fold)[:, 1]
        y_val_pred = (y_val_proba >= 0.67).astype(int)
        
        acc = accuracy_score(y_val_fold, y_val_pred)
        prec = precision_score(y_val_fold, y_val_pred, zero_division=0)
        rec = recall_score(y_val_fold, y_val_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val_fold, y_val_proba) if len(np.unique(y_val_fold)) > 1 else float('nan')
        
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_roc_aucs.append(roc_auc)
        
        log.info(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Summary
    log.info("\n" + "="*80)
    log.info("CROSS-VALIDATION RESULTS")
    log.info("="*80)
    log.info(f"Accuracy:  {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    log.info(f"Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    log.info(f"Recall:    {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    log.info(f"F1:        {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    log.info(f"ROC-AUC:   {np.mean(fold_roc_aucs):.4f} ± {np.std(fold_roc_aucs):.4f}" if not np.isnan(np.mean(fold_roc_aucs)) else "ROC-AUC:   N/A")
    
    # Save results
    results = {
        "n_folds": args.n_folds,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_precision": float(np.mean(fold_precisions)),
        "mean_recall": float(np.mean(fold_recalls)),
        "mean_f1": float(np.mean(fold_f1s)),
        "mean_roc_auc": float(np.mean(fold_roc_aucs)) if not np.isnan(np.mean(fold_roc_aucs)) else None,
        "fold_results": {
            "accuracies": [float(a) for a in fold_accuracies],
            "precisions": [float(p) for p in fold_precisions],
            "recalls": [float(r) for r in fold_recalls],
            "f1s": [float(f) for f in fold_f1s],
            "roc_aucs": [float(r) for r in fold_roc_aucs if not np.isnan(r)],
        },
    }
    
    results_path = cache_dir / "cross_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


