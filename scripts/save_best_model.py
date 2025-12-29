#!/usr/bin/env python3
"""Save the best model (feature interactions) to the default model path.

Usage:
    poetry run python scripts/save_best_model.py [IMAGE_DIR]
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.model_version import create_model_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("save_best")


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
    n_base = X.shape[1]
    n_select = min(top_n, n_base)
    
    # Select top features by variance
    feature_vars = np.var(X, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:n_select]
    X_selected = X[:, top_indices]
    
    # Create pairwise interactions
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
        X_final = np.hstack([X, X_interactions])
        return X_final, interaction_pairs
    
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
        X_final = np.hstack([X, X_ratios])
        return X_final, ratio_pairs
    
    return X, []


class FeatureInteractionTransformer:
    """Transformer to add feature interactions during inference."""
    
    def __init__(self, interaction_pairs: list[tuple[int, int]], ratio_pairs: list[tuple[int, int]] = None):
        self.interaction_pairs = interaction_pairs
        self.ratio_pairs = ratio_pairs or []
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Add interactions and ratios to feature matrix."""
        X_out = X.copy()
        
        # Add interactions
        interactions = []
        for i, j in self.interaction_pairs:
            interaction = X[:, i] * X[:, j]
            interactions.append(interaction)
        
        if interactions:
            X_out = np.hstack([X_out, np.column_stack(interactions)])
        
        # Add ratios
        ratios = []
        for i, j in self.ratio_pairs:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(X[:, i], X[:, j], 
                                out=np.zeros_like(X[:, i]), 
                                where=X[:, j]!=0)
                ratios.append(ratio)
        
        if ratios:
            X_out = np.hstack([X_out, np.column_stack(ratios)])
        
        return X_out


def main():
    parser = argparse.ArgumentParser(description="Save best model to default path")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension for embeddings")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    # Find embeddings file
    embeddings_path = args.embeddings
    if not embeddings_path:
        cache_dir = Path(__file__).resolve().parent.parent / ".cache"
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
    
    log.info("="*80)
    log.info("SAVING BEST MODEL (Feature Interactions)")
    log.info("="*80)
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info(f"Loading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} handcrafted features")
    
    # Load embeddings
    log.info(f"Loading embeddings from {embeddings_path}...")
    emb, emb_fnames = load_embeddings(embeddings_path)
    log.info(f"Embeddings: {emb.shape[1]} dimensions for {len(emb_fnames)} images")
    
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
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load tuned CatBoost hyperparameters
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    tuning_path = cache_dir / "catboost_tuning_results.json"
    
    if tuning_path.exists():
        with open(tuning_path) as f:
            tuning_data = json.load(f)
            cb_params = tuning_data["best_params"]
    else:
        cb_params = {
            "iterations": 500,
            "learning_rate": 0.07,
            "depth": 9,
            "l2_leaf_reg": 2.48,
            "border_count": 81,
        }
    
    # Train CatBoost
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return 1
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params_final = {
        **cb_params,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params_final)),
    ])
    
    log.info("\nTraining CatBoost with feature interactions...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.67).astype(int)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    log.info("\n" + "="*80)
    log.info("MODEL EVALUATION")
    log.info("="*80)
    log.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1:        {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Create feature interaction transformer
    feature_transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    
    # Save model
    from src.training_core import DEFAULT_MODEL_PATH
    model_path = DEFAULT_MODEL_PATH
    
    metadata = create_model_metadata(
        feature_count=X.shape[1],  # Handcrafted features only
        feature_mode="FULL",
        params=cb_params_final,
        n_samples=len(y_train),
    )
    metadata["model_type"] = "CatBoost"
    metadata["has_embeddings"] = True
    metadata["n_embedding_features"] = int(emb.shape[1])
    metadata["has_feature_interactions"] = True
    metadata["n_interaction_features"] = len(interaction_pairs)
    metadata["n_ratio_features"] = len(ratio_pairs)
    metadata["n_base_features"] = int(X_combined.shape[1])
    metadata["n_total_features"] = int(X_final.shape[1])
    metadata["optimal_threshold"] = 0.67
    metadata["test_accuracy"] = float(accuracy)
    metadata["test_precision"] = float(precision)
    metadata["test_recall"] = float(recall)
    metadata["test_f1"] = float(f1)
    metadata["test_roc_auc"] = float(roc_auc) if not np.isnan(roc_auc) else None
    
    model_data = {
        "__metadata__": metadata,
        "model": clf,
        "pca": pca,
        "feature_transformer": feature_transformer,  # For adding interactions
        "feature_length": int(X_final.shape[1]),  # Total features after interactions
        "n_base_features": int(X_combined.shape[1]),  # Base features (before interactions)
        "n_samples": len(y_train),
        "n_keep": n_keep,
        "n_trash": n_trash,
        "precision": float(precision),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "f1": float(f1),
        "optimal_threshold": 0.67,
    }
    
    joblib.dump(model_data, model_path)
    log.info(f"\nModel saved to: {model_path}")
    log.info(f"  Accuracy: {accuracy*100:.2f}%")
    log.info(f"  Features: {X_final.shape[1]} total ({X_combined.shape[1]} base + {len(interaction_pairs)} interactions + {len(ratio_pairs)} ratios)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
