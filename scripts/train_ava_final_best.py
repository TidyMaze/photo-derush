#!/usr/bin/env python3
"""Final best model - trying all remaining improvements.

Usage:
    poetry run python scripts/train_ava_final_best.py [--max-ava N]
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
log = logging.getLogger("train_ava_final")


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
        log.error(f"AVA metadata not found")
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
            image_scores[image_id] = {'class': rounded_score - 1, 'mean_score': mean_score}
    return image_scores


def prepare_data(cache_dir: Path, max_ava: int = None, pca_dim: int = 128):
    """Prepare and return all data."""
    ava_features_path = cache_dir / "ava_features.joblib"
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
    if max_ava and len(X_ava) > max_ava:
        X_ava = X_ava[:max_ava]
        ava_ids = ava_ids[:max_ava]
    
    image_scores = load_ava_multiclass_labels(cache_dir)
    if image_scores is None:
        return None
    
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
    
    X_ava = X_ava[valid_indices]
    y_ava = np.array(y_ava)
    
    embeddings_path = None
    for p in [cache_dir / "embeddings_resnet18_full.joblib", cache_dir / "embeddings_resnet18.joblib"]:
        if p.exists():
            embeddings_path = str(p)
            break
    
    X = X_ava
    y = y_ava
    filenames = [f"ava_{i}" for i in range(len(y_ava))]
    
    if embeddings_path:
        emb, emb_fnames = load_embeddings(embeddings_path)
        if pca_dim and pca_dim < emb.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
        X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    else:
        X_combined = X
    
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
    
    return X_final, y


def train_and_evaluate(config_name: str, X_train, X_val, X_test, y_train, y_val, y_test, params: dict):
    """Train model and return results."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    
    cb_params = {
        **params,
        "bootstrap_type": "Bernoulli",
        "random_seed": 42,
        "verbose": False,
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
    
    cat_model = clf.named_steps['cat']
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_val_pred = cat_model.predict(X_val_scaled).flatten().astype(int)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    return {
        "config": config_name,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "model": clf,
    }


def main():
    parser = argparse.ArgumentParser(description="Final best model - all improvements")
    parser.add_argument("--max-ava", type=int, default=None)
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("FINAL BEST MODEL - TRYING ALL REMAINING IMPROVEMENTS")
    log.info("="*80)
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Try different PCA dimensions and feature configurations
    improvements = []
    
    # Improvement 1: Different PCA dimensions
    log.info("\n=== Improvement 1: Different PCA dimensions ===")
    for pca_dim in [64, 128, 192, 256]:
        log.info(f"Trying PCA dimension: {pca_dim}")
        data = prepare_data(cache_dir, args.max_ava, pca_dim)
        if data is None:
            continue
        X_final, y = data
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        
        params = {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "rsm": 0.88,
            "subsample": 0.85,
        }
        
        result = train_and_evaluate(f"pca_{pca_dim}", X_train, X_val, X_test, y_train, y_val, y_test, params)
        if result:
            improvements.append(result)
            log.info(f"  PCA {pca_dim}: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Improvement 2: Best config from previous with more iterations
    log.info("\n=== Improvement 2: Extended training ===")
    data = prepare_data(cache_dir, args.max_ava, 128)
    if data:
        X_final, y = data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        
        configs = [
            ("extended_3000", {"iterations": 3000, "learning_rate": 0.015, "depth": 7, "l2_leaf_reg": 3.0, "rsm": 0.88, "subsample": 0.85}),
            ("extended_4000", {"iterations": 4000, "learning_rate": 0.012, "depth": 7, "l2_leaf_reg": 3.0, "rsm": 0.88, "subsample": 0.85}),
            ("fine_tuned", {"iterations": 2000, "learning_rate": 0.020, "depth": 7, "l2_leaf_reg": 2.8, "rsm": 0.90, "subsample": 0.87}),
        ]
        
        for name, params in configs:
            log.info(f"Trying {name}...")
            result = train_and_evaluate(name, X_train, X_val, X_test, y_train, y_val, y_test, params)
            if result:
                improvements.append(result)
                log.info(f"  {name}: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Find best
    if not improvements:
        log.error("No successful improvements")
        return 1
    
    best = max(improvements, key=lambda r: r['val_acc'])
    
    log.info("\n" + "="*80)
    log.info("BEST FINAL MODEL")
    log.info("="*80)
    log.info(f"Configuration: {best['config']}")
    log.info(f"Validation accuracy: {best['val_acc']:.4f} ({best['val_acc']*100:.2f}%)")
    log.info(f"Test accuracy: {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)")
    
    # Final evaluation
    data = prepare_data(cache_dir, args.max_ava, 128)
    if data:
        X_final, y = data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        
        # Retrain best model for final evaluation
        best_params = {
            "iterations": 2500,
            "learning_rate": 0.018,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "rsm": 0.88,
            "subsample": 0.85,
        }
        
        final_result = train_and_evaluate("final_best", X_train, X_val, X_test, y_train, y_val, y_test, best_params)
        
        if final_result:
            model = final_result['model']
            cat_model = model.named_steps['cat']
            X_test_scaled = model.named_steps['scaler'].transform(X_test)
            y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
            
            unique_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
            target_names = [f"Score {i+1}" for i in unique_classes]
            
            log.info("\nFinal Classification Report:")
            log.info(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names, zero_division=0))
            
            # Save
            model_path = cache_dir / "catboost_ava_multiclass_final_best.joblib"
            joblib.dump(model, model_path)
            log.info(f"\nFinal best model saved to: {model_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

