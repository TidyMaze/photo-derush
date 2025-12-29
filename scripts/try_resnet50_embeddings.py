#!/usr/bin/env python3
"""Try ResNet50 embeddings instead of ResNet18 for better performance.

Usage:
    poetry run python scripts/try_resnet50_embeddings.py [IMAGE_DIR] [--pca-dim 256]
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("resnet50")


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


def main():
    parser = argparse.ArgumentParser(description="Try ResNet50 embeddings")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to ResNet50 embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=256, help="PCA dimension for embeddings")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    # Find embeddings file
    embeddings_path = args.embeddings
    if not embeddings_path:
        cache_dir = Path(__file__).resolve().parent.parent / ".cache"
        possible = [
            cache_dir / "embeddings_resnet50.joblib",
            cache_dir / "embeddings_resnet50_full.joblib",
        ]
        for p in possible:
            if p.exists():
                embeddings_path = str(p)
                break
    
    if not embeddings_path or not os.path.exists(embeddings_path):
        log.error("No ResNet50 embeddings found. Build them first:")
        log.error("  poetry run python scripts/build_cnn_embeddings.py ~/Pictures/photo-dataset --model resnet50")
        return 1
    
    log.info("="*80)
    log.info("TESTING RESNET50 EMBEDDINGS")
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
    log.info(f"Loading ResNet50 embeddings from {embeddings_path}...")
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
    log.info(f"Combined features: {X_combined.shape[1]} total ({X.shape[1]} handcrafted + {emb.shape[1]} embeddings)")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, stratify=y, random_state=42
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
    
    log.info(f"\nCatBoost hyperparameters: {cb_params}")
    
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
    
    log.info("\nTraining CatBoost with ResNet50 embeddings...")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.67).astype(int)  # Use optimal threshold
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    log.info("\n" + "="*80)
    log.info("RESNET50 EMBEDDINGS RESULTS")
    log.info("="*80)
    log.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1:        {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Compare with ResNet18 baseline
    baseline_path = cache_dir / "embeddings_training_results.json"
    baseline_acc = 0.88
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
            baseline_acc = baseline_data["test_metrics"]["accuracy"]
    
    improvement = (accuracy - baseline_acc) * 100
    log.info(f"\nComparison:")
    log.info(f"  ResNet18 baseline: {baseline_acc*100:.2f}%")
    log.info(f"  ResNet50 result:  {accuracy*100:.2f}%")
    log.info(f"  Improvement:      {improvement:+.2f} percentage points")
    
    # Save results
    results_path = cache_dir / "resnet50_embeddings_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model_type": "CatBoost",
            "embedding_model": "ResNet50",
            "n_handcrafted_features": int(X.shape[1]),
            "n_embedding_features": int(emb.shape[1]),
            "n_total_features": int(X_combined.shape[1]),
            "pca_dim": args.pca_dim if pca else None,
            "test_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
            "improvement_vs_resnet18": float(improvement),
        }, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

