#!/usr/bin/env python3
"""Train improved multi-class model on AVA dataset - trying all improvements.

Usage:
    poetry run python scripts/train_ava_multiclass_improved.py [--max-ava N]
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("train_ava_improved")


def load_embeddings(path: str):
    """Load embeddings from joblib file."""
    data = joblib.load(path)
    return data['embeddings'], data.get('filenames', [])


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


def load_ava_multiclass_labels(cache_dir: Path):
    """Load AVA labels with full score distribution (1-10)."""
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
        log.error(f"AVA metadata not found at {ava_metadata_path}")
        return None
    
    image_scores = {}
    
    with open(ava_metadata_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            image_id = parts[1]
            score_counts = [int(s) for s in parts[2:12]]
            total_votes = sum(score_counts)
            if total_votes == 0:
                continue
            
            weighted_sum = sum((i+1) * count for i, count in enumerate(score_counts))
            mean_score = weighted_sum / total_votes
            rounded_score = int(round(mean_score))
            rounded_score = max(1, min(10, rounded_score))
            
            image_scores[image_id] = {
                'class': rounded_score - 1,  # 0-9 for sklearn
                'mean_score': mean_score,
            }
    
    return image_scores


def try_configuration(config_name: str, X_train, X_val, X_test, y_train, y_val, y_test, best_params: dict):
    """Try a specific configuration and return results."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return None
    
    # Different configurations to try
    configs = {
        "config1_high_iter": {
            "iterations": 3000,
            "learning_rate": 0.015,
            "depth": 7,
            "l2_leaf_reg": 4.0,
            "rsm": 0.85,
            "subsample": 0.83,
        },
        "config2_moderate": {
            "iterations": 2000,
            "learning_rate": 0.023,
            "depth": 7,
            "l2_leaf_reg": 5.3,
            "rsm": 0.81,
            "subsample": 0.79,
        },
        "config3_deeper": {
            "iterations": 2000,
            "learning_rate": 0.02,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "rsm": 0.80,
            "subsample": 0.80,
        },
        "config4_tuned_based": {
            "iterations": 1500,
            "learning_rate": best_params.get("learning_rate", 0.038),
            "depth": best_params.get("depth", 8),
            "l2_leaf_reg": best_params.get("l2_leaf_reg", 3.5),
            "rsm": best_params.get("rsm", 0.85),
            "subsample": best_params.get("subsample", 0.83),
        },
        "config5_low_reg": {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "rsm": 0.88,
            "subsample": 0.85,
        },
        "config6_balanced": {
            "iterations": 1800,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 4.5,
            "rsm": 0.83,
            "subsample": 0.82,
        },
    }
    
    if config_name not in configs:
        log.error(f"Unknown configuration: {config_name}")
        return None
    
    params = configs[config_name]
    
    cb_params = {
        **params,
        "bootstrap_type": "Bernoulli",
        "random_seed": 42,
        "verbose": False,  # Less verbose for multiple configs
        "thread_count": -1,
        "loss_function": "MultiClass",
        "classes_count": 10,
        "early_stopping_rounds": 200,
        "eval_metric": "Accuracy",
        "use_best_model": True,
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    # Evaluate
    cat_model = clf.named_steps['cat']
    X_train_scaled = clf.named_steps['scaler'].transform(X_train)
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_train_pred = cat_model.predict(X_train_scaled).flatten().astype(int)
    y_val_pred = cat_model.predict(X_val_scaled).flatten().astype(int)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    return {
        "config": config_name,
        "params": params,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "overfitting": train_acc - val_acc,
        "model": clf,
    }


def main():
    parser = argparse.ArgumentParser(description="Train improved multi-class model on AVA dataset")
    parser.add_argument("--max-ava", type=int, default=None, help="Max AVA samples")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimensions")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set size")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("IMPROVED MULTI-CLASS TRAINING - TRYING ALL CONFIGURATIONS")
    log.info("="*80)
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Load tuned hyperparameters
    tuning_results_path = cache_dir / "catboost_ava_tuning_results.json"
    best_params = {}
    if tuning_results_path.exists():
        with open(tuning_results_path) as f:
            tuning_results = json.load(f)
        best_params = tuning_results['best_params']
        log.info(f"Using tuned hyperparameters as baseline")
    
    # Load AVA features
    ava_features_path = cache_dir / "ava_features.joblib"
    if not ava_features_path.exists():
        log.error(f"AVA features not found at {ava_features_path}")
        return 1
    
    log.info("Loading AVA features...")
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
    if args.max_ava and len(X_ava) > args.max_ava:
        X_ava = X_ava[:args.max_ava]
        ava_ids = ava_ids[:args.max_ava]
    
    log.info(f"AVA features: {len(X_ava)} samples, {X_ava.shape[1]} features")
    
    # Load multi-class labels
    image_scores = load_ava_multiclass_labels(cache_dir)
    if image_scores is None:
        return 1
    
    # Match features to labels
    y_ava = []
    valid_indices = []
    for i, img_id in enumerate(ava_ids):
        img_id_str = str(img_id)
        if img_id_str in image_scores:
            y_ava.append(image_scores[img_id_str]['class'])
            valid_indices.append(i)
        elif f"{img_id}.jpg" in image_scores:
            y_ava.append(image_scores[f"{img_id}.jpg"]['class'])
            valid_indices.append(i)
    
    if len(y_ava) == 0:
        ava_labeled_path = cache_dir / "ava_dataset" / "ava_keep_trash_labels.json"
        if ava_labeled_path.exists():
            with open(ava_labeled_path) as f:
                labeled = json.load(f)
            y_ava = []
            for i in range(len(X_ava)):
                if i < len(labeled):
                    mean_score = labeled[i].get('score', 5.0)
                    rounded = int(round(mean_score))
                    rounded = max(1, min(10, rounded))
                    y_ava.append(rounded - 1)
                else:
                    y_ava.append(4)
            valid_indices = list(range(len(X_ava)))
        else:
            log.error("No labels found")
            return 1
    
    X_ava = X_ava[valid_indices]
    y_ava = np.array(y_ava)
    
    log.info(f"Matched {len(y_ava)} samples with labels")
    
    # Load embeddings
    embeddings_path = None
    possible = [
        cache_dir / "embeddings_resnet18_full.joblib",
        cache_dir / "embeddings_resnet18.joblib",
    ]
    for p in possible:
        if p.exists():
            embeddings_path = str(p)
            break
    
    X = X_ava
    y = y_ava
    filenames = [f"ava_{i}" for i in range(len(y_ava))]
    
    if embeddings_path and os.path.exists(embeddings_path):
        log.info(f"Loading embeddings from {embeddings_path}...")
        emb, emb_fnames = load_embeddings(embeddings_path)
        
        if args.pca_dim and args.pca_dim < emb.shape[1]:
            log.info(f"Applying PCA to reduce embeddings to {args.pca_dim} dimensions...")
            pca = PCA(n_components=args.pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
            log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        X_combined = align_and_concat(X, filenames, emb, emb_fnames)
        log.info(f"Base features: {X_combined.shape[1]} total ({X.shape[1]} handcrafted + {emb.shape[1]} embeddings)")
    else:
        log.warning("No embeddings found, using handcrafted features only")
        X_combined = X
    
    # Add feature interactions
    log.info("Adding feature interactions...")
    top_n = 15
    feature_vars = np.var(X_combined, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:top_n]
    
    interaction_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < 100:
                interaction_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    ratio_pairs = []
    for i in range(min(20, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 5, len(top_indices))):
            if len(ratio_pairs) < 20:
                ratio_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    log.info(f"Final feature count: {X_final.shape[1]}")
    
    # Split: train / validation / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y, test_size=args.test_size, stratify=y, random_state=42
    )
    
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, stratify=y_trainval, random_state=42
    )
    
    log.info(f"\nDataset splits:")
    log.info(f"  Train: {len(X_train)} samples")
    log.info(f"  Validation: {len(X_val)} samples")
    log.info(f"  Test: {len(X_test)} samples")
    
    # Try all configurations
    log.info("\n" + "="*80)
    log.info("TESTING ALL CONFIGURATIONS")
    log.info("="*80)
    
    results = []
    configs_to_try = [
        "config1_high_iter",
        "config2_moderate",
        "config3_deeper",
        "config4_tuned_based",
        "config5_low_reg",
        "config6_balanced",
    ]
    
    for config_name in configs_to_try:
        log.info(f"\nTrying {config_name}...")
        result = try_configuration(config_name, X_train, X_val, X_test, y_train, y_val, y_test, best_params)
        if result:
            results.append(result)
            log.info(f"  Validation accuracy: {result['val_acc']:.4f} ({result['val_acc']*100:.2f}%)")
            log.info(f"  Test accuracy: {result['test_acc']:.4f} ({result['test_acc']*100:.2f}%)")
            log.info(f"  Overfitting gap: {result['overfitting']:.4f}")
    
    # Find best configuration
    if not results:
        log.error("No successful configurations")
        return 1
    
    best_result = max(results, key=lambda r: r['val_acc'])
    
    log.info("\n" + "="*80)
    log.info("BEST CONFIGURATION")
    log.info("="*80)
    log.info(f"Configuration: {best_result['config']}")
    log.info(f"Parameters: {best_result['params']}")
    log.info(f"Train accuracy: {best_result['train_acc']:.4f} ({best_result['train_acc']*100:.2f}%)")
    log.info(f"Validation accuracy: {best_result['val_acc']:.4f} ({best_result['val_acc']*100:.2f}%)")
    log.info(f"Test accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc']*100:.2f}%)")
    log.info(f"Overfitting gap: {best_result['overfitting']:.4f} ({best_result['overfitting']*100:.2f}%)")
    
    # Final evaluation with best model
    log.info("\n" + "="*80)
    log.info("FINAL EVALUATION")
    log.info("="*80)
    
    best_model = best_result['model']
    cat_model = best_model.named_steps['cat']
    X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    target_names = [f"Score {i+1}" for i in unique_classes]
    
    log.info("\nClassification Report (Test Set):")
    log.info(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names, zero_division=0))
    
    log.info("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred, labels=unique_classes)
    log.info(f"\n{cm}")
    
    # Save best model
    model_path = cache_dir / "catboost_ava_multiclass_best.joblib"
    joblib.dump(best_model, model_path)
    log.info(f"\nBest model saved to: {model_path}")
    
    # Save results
    results_path = cache_dir / "ava_multiclass_improved_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_config": best_result['config'],
            "best_params": best_result['params'],
            "best_train_acc": float(best_result['train_acc']),
            "best_val_acc": float(best_result['val_acc']),
            "best_test_acc": float(best_result['test_acc']),
            "best_overfitting": float(best_result['overfitting']),
            "all_results": [
                {
                    "config": r['config'],
                    "val_acc": float(r['val_acc']),
                    "test_acc": float(r['test_acc']),
                    "overfitting": float(r['overfitting']),
                }
                for r in results
            ],
        }, f, indent=2)
    
    log.info(f"Results saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

