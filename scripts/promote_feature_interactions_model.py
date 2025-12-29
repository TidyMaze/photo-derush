#!/usr/bin/env python3
"""Promote feature interactions model to main model by retraining and saving.

This is a simpler version that retrains the model and saves it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.model_version import create_model_metadata
from src.training_core import DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("promote")


def load_embeddings(path: str):
    data = joblib.load(path)
    return data['embeddings'], data['filenames']


def align_and_concat(X_feats: np.ndarray, filenames: list[str], emb: np.ndarray, emb_fnames: list[str]):
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


class FeatureInteractionTransformer:
    def __init__(self, interaction_pairs: list[tuple[int, int]], ratio_pairs: list[tuple[int, int]] = None):
        self.interaction_pairs = interaction_pairs
        self.ratio_pairs = ratio_pairs or []
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = X.copy()
        interactions = [X[:, i] * X[:, j] for i, j in self.interaction_pairs]
        if interactions:
            X_out = np.hstack([X_out, np.column_stack(interactions)])
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
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    embeddings_path = cache_dir / "embeddings_resnet18.joblib"
    if not embeddings_path.exists():
        log.error("No embeddings found")
        return 1
    
    log.info("Loading dataset...")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    if not repo:
        log.error("No repository found")
        return 1
    
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    emb, emb_fnames = load_embeddings(str(embeddings_path))
    pca = PCA(n_components=128, random_state=42)
    emb = pca.fit_transform(emb)
    
    X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    X_final, interaction_pairs = add_interactions(X_combined, top_n=15, max_interactions=100)
    X_final, ratio_pairs = add_ratio_features(X_final, top_indices=list(range(min(10, X.shape[1]))), n_ratios=20)
    
    log.info(f"Features: {X.shape[1]} handcrafted + {emb.shape[1]} embeddings = {X_combined.shape[1]} base → {X_final.shape[1]} total")
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)
    
    tuning_path = cache_dir / "catboost_tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path) as f:
            cb_params = json.load(f)["best_params"]
    else:
        cb_params = {"iterations": 500, "learning_rate": 0.07, "depth": 9, "l2_leaf_reg": 2.48, "border_count": 81}
    
    from catboost import CatBoostClassifier
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**{**cb_params, "scale_pos_weight": scale_pos_weight, "random_seed": 42, "verbose": False, "thread_count": -1})),
    ])
    
    log.info("Training...")
    clf.fit(X_train, y_train)
    
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.67).astype(int)
    accuracy = accuracy_score(y_test, y_test_pred)
    
    log.info(f"Test accuracy: {accuracy*100:.2f}%")
    
    feature_transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    
    metadata = create_model_metadata(
        feature_count=X.shape[1],
        feature_mode="FULL",
        params={**cb_params, "scale_pos_weight": scale_pos_weight},
        n_samples=len(y_train),
    )
    metadata.update({
        "model_type": "CatBoost",
        "has_embeddings": True,
        "n_embedding_features": 128,
        "has_feature_interactions": True,
        "n_interaction_features": len(interaction_pairs),
        "n_ratio_features": len(ratio_pairs),
        "n_base_features": int(X_combined.shape[1]),
        "n_total_features": int(X_final.shape[1]),
        "optimal_threshold": 0.67,
        "test_accuracy": float(accuracy),
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
    log.info(f"Model saved to: {DEFAULT_MODEL_PATH}")
    log.info(f"Accuracy: {accuracy*100:.2f}%")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


