#!/usr/bin/env python3
"""Train final combined model with all improvements: tuned CatBoost + embeddings + optimal threshold.

Usage:
    poetry run python scripts/train_final_combined_model.py [IMAGE_DIR]
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
log = logging.getLogger("train_final")


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


def load_tuned_params():
    """Load tuned CatBoost hyperparameters."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    tuning_path = cache_dir / "catboost_tuning_results.json"
    
    if tuning_path.exists():
        with open(tuning_path) as f:
            data = json.load(f)
            return data["best_params"]
    
    # Fallback to default optimal params
    return {
        "iterations": 200,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "border_count": 128,
    }


def load_optimal_threshold():
    """Load optimal threshold from threshold optimization."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    threshold_path = cache_dir / "threshold_optimization_results.json"
    
    if threshold_path.exists():
        with open(threshold_path) as f:
            data = json.load(f)
            return data["optimal_threshold"]
    
    return 0.5  # Default


def main():
    parser = argparse.ArgumentParser(description="Train final combined model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings joblib")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension for embeddings")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
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
    log.info("TRAINING FINAL COMBINED MODEL")
    log.info("="*80)
    log.info("Improvements applied:")
    log.info("  - Tuned CatBoost hyperparameters")
    log.info("  - ResNet18 embeddings (128 dims)")
    log.info("  - Optimal decision threshold")
    
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
    log.info(f"Combined features: {X_combined.shape[1]} total ({X.shape[1]} handcrafted + {emb.shape[1]} embeddings)")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Split train for threshold optimization
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    log.info(f"Train: {len(y_train_split)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    
    # Load tuned hyperparameters
    cb_params = load_tuned_params()
    log.info(f"\nCatBoost hyperparameters: {cb_params}")
    
    # Train CatBoost
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return 1
    
    n_keep = int(np.sum(y_train_split == 1))
    n_trash = int(np.sum(y_train_split == 0))
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
    
    log.info("\nTraining CatBoost with all improvements...")
    clf.fit(X_train_split, y_train_split)
    
    # Optimize threshold on validation set
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.3, 0.71, 0.01)
    best_threshold = 0.5
    best_val_acc = 0.0
    
    for threshold in thresholds:
        y_val_pred = (y_val_proba >= threshold).astype(int)
        acc = accuracy_score(y_val, y_val_pred)
        if acc > best_val_acc:
            best_val_acc = acc
            best_threshold = threshold
    
    log.info(f"\nOptimal threshold: {best_threshold:.3f} (validation accuracy: {best_val_acc:.2%})")
    
    # Evaluate on test set with optimal threshold
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    log.info("\n" + "="*80)
    log.info("FINAL TEST SET RESULTS")
    log.info("="*80)
    log.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1:        {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    log.info(f"\nThreshold: {best_threshold:.3f}")
    
    # Compare with baseline
    baseline_acc = 0.72
    improvement = (accuracy - baseline_acc) * 100
    log.info(f"\nBaseline (XGBoost): 72.00%")
    log.info(f"Final model: {accuracy*100:.2f}%")
    log.info(f"Total improvement: +{improvement:.2f} percentage points")
    
    # Save model
    model_path = os.path.expanduser("~/.photo-derush-final-combined-model.joblib")
    
    from src.model_version import create_model_metadata
    from src.features import FEATURE_COUNT, USE_FULL_FEATURES
    
    metadata = create_model_metadata(
        feature_count=FEATURE_COUNT,
        feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
        params=cb_params_final,
        n_samples=len(y_train_split),
    )
    metadata["model_type"] = "CatBoost"
    metadata["has_embeddings"] = True
    metadata["n_embedding_features"] = int(emb.shape[1])
    metadata["optimal_threshold"] = float(best_threshold)
    metadata["test_accuracy"] = float(accuracy)
    metadata["test_precision"] = float(precision)
    metadata["test_recall"] = float(recall)
    metadata["test_f1"] = float(f1)
    metadata["test_roc_auc"] = float(roc_auc) if not np.isnan(roc_auc) else None
    
    model_data = {
        "__metadata__": metadata,
        "model": clf,
        "pca": pca,
        "feature_length": X_combined.shape[1],
        "n_samples": len(y_train_split),
        "n_keep": n_keep,
        "n_trash": n_trash,
        "precision": float(precision),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "f1": float(f1),
        "optimal_threshold": float(best_threshold),
    }
    
    joblib.dump(model_data, model_path)
    log.info(f"\nModel saved to: {model_path}")
    
    # Save results
    results_path = Path(__file__).resolve().parent.parent / ".cache" / "final_combined_model_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model_type": "CatBoost",
            "hyperparameters": cb_params_final,
            "n_handcrafted_features": int(X.shape[1]),
            "n_embedding_features": int(emb.shape[1]),
            "n_total_features": int(X_combined.shape[1]),
            "optimal_threshold": float(best_threshold),
            "test_metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            },
            "improvement_vs_baseline": float(improvement),
        }, f, indent=2)
    
    log.info(f"Results saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

