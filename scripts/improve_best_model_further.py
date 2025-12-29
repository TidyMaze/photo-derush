#!/usr/bin/env python3
"""Try additional improvements to push beyond 48.88% validation accuracy.

Improvements to try:
1. Use full dataset (not limited to 10k)
2. Better feature engineering (more interactions)
3. Ensemble methods
4. Different embedding dimensions
5. Feature selection
6. Data augmentation strategies
7. More hyperparameter tuning
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("improve_best")


def load_embeddings(path: str):
    data = joblib.load(path)
    return data['embeddings'], data.get('filenames', [])


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


def load_ava_multiclass_labels(cache_dir: Path):
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
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
            image_scores[image_id] = {'class': rounded_score - 1}
    return image_scores


def prepare_data(cache_dir: Path, max_ava: int = None, pca_dim: int = 128, more_interactions: bool = False):
    """Prepare data with optional improvements."""
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
    
    # Feature interactions
    top_n = 20 if more_interactions else 15
    max_interactions = 150 if more_interactions else 100
    feature_vars = np.var(X_combined, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:top_n]
    
    interaction_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < max_interactions:
                interaction_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    ratio_pairs = []
    for i in range(min(25 if more_interactions else 20, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 7 if more_interactions else 5, len(top_indices))):
            if len(ratio_pairs) < 30 if more_interactions else 20:
                ratio_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    return X_final, y


def train_and_evaluate(name: str, X_train, X_val, X_test, y_train, y_val, y_test, params: dict, use_feature_selection: bool = False):
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
    
    steps = [("scaler", StandardScaler())]
    
    # Optional feature selection
    if use_feature_selection:
        # Select top features (avoid constant features)
        k = min(200, X_train.shape[1])
        # Remove constant features first
        non_constant = []
        for i in range(X_train.shape[1]):
            if np.std(X_train[:, i]) > 1e-10:
                non_constant.append(i)
        k = min(k, len(non_constant))
        if k > 0:
            steps.append(("selector", SelectKBest(f_classif, k=k)))
    
    steps.append(("cat", CatBoostClassifier(**cb_params)))
    
    clf = Pipeline(steps)
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    cat_model = clf.named_steps['cat']
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    if use_feature_selection:
        X_val_scaled = clf.named_steps['selector'].transform(X_val_scaled)
        X_test_scaled = clf.named_steps['selector'].transform(X_test_scaled)
    
    y_val_pred = cat_model.predict(X_val_scaled).flatten().astype(int)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    return {
        "name": name,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "model": clf,
    }


def main():
    parser = argparse.ArgumentParser(description="Try additional improvements")
    parser.add_argument("--max-ava", type=int, default=None, help="Max samples (None = all)")
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("ADDITIONAL IMPROVEMENTS TO BEST MODEL")
    log.info("="*80)
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Best params from previous
    best_params = {
        "iterations": 2500,
        "learning_rate": 0.018,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "rsm": 0.88,
        "subsample": 0.85,
    }
    
    improvements = []
    
    # Improvement 1: Use full dataset
    log.info("\n=== Improvement 1: Full Dataset (no limit) ===")
    data = prepare_data(cache_dir, max_ava=None, pca_dim=128, more_interactions=False)
    if data:
        X_final, y = data
        log.info(f"Full dataset: {len(y)} samples")
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        
        result = train_and_evaluate("full_dataset", X_train, X_val, X_test, y_train, y_val, y_test, best_params)
        if result:
            improvements.append(result)
            log.info(f"  Full dataset: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Improvement 2: More feature interactions
    log.info("\n=== Improvement 2: More Feature Interactions ===")
    data = prepare_data(cache_dir, max_ava=args.max_ava, pca_dim=128, more_interactions=True)
    if data:
        X_final, y = data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        log.info(f"  Features: {X_final.shape[1]} (with more interactions)")
        
        result = train_and_evaluate("more_interactions", X_train, X_val, X_test, y_train, y_val, y_test, best_params)
        if result:
            improvements.append(result)
            log.info(f"  More interactions: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Improvement 3: Feature selection (skipped - causes issues with CatBoost)
    # log.info("\n=== Improvement 3: Feature Selection ===")
    # Skipped due to CatBoost compatibility issues
    
    # Improvement 4: Optimized hyperparameters
    log.info("\n=== Improvement 4: Optimized Hyperparameters ===")
    data = prepare_data(cache_dir, max_ava=args.max_ava, pca_dim=128, more_interactions=False)
    if data:
        X_final, y = data
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_final, y, test_size=0.2, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
        )
        
        optimized_params = {
            "iterations": 3000,
            "learning_rate": 0.016,
            "depth": 7,
            "l2_leaf_reg": 2.8,
            "rsm": 0.90,
            "subsample": 0.87,
        }
        
        result = train_and_evaluate("optimized_params", X_train, X_val, X_test, y_train, y_val, y_test, optimized_params)
        if result:
            improvements.append(result)
            log.info(f"  Optimized params: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Improvement 5: Higher PCA dimensions
    log.info("\n=== Improvement 5: Higher PCA Dimensions ===")
    for pca_dim in [192, 256]:
        data = prepare_data(cache_dir, max_ava=args.max_ava, pca_dim=pca_dim, more_interactions=False)
        if data:
            X_final, y = data
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X_final, y, test_size=0.2, stratify=y, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
            )
            
            result = train_and_evaluate(f"pca_{pca_dim}", X_train, X_val, X_test, y_train, y_val, y_test, best_params)
            if result:
                improvements.append(result)
                log.info(f"  PCA {pca_dim}: Val={result['val_acc']:.4f}, Test={result['test_acc']:.4f}")
    
    # Find best
    if not improvements:
        log.error("No successful improvements")
        return 1
    
    best = max(improvements, key=lambda r: r['val_acc'])
    
    log.info("\n" + "="*80)
    log.info("BEST IMPROVEMENT")
    log.info("="*80)
    log.info(f"Configuration: {best['name']}")
    log.info(f"Validation accuracy: {best['val_acc']:.4f} ({best['val_acc']*100:.2f}%)")
    log.info(f"Test accuracy: {best['test_acc']:.4f} ({best['test_acc']*100:.2f}%)")
    
    # Save best improved model
    model_path = cache_dir / "catboost_ava_multiclass_improved_further.joblib"
    joblib.dump(best['model'], model_path)
    log.info(f"\nImproved model saved to: {model_path}")
    
    # Save results
    results_path = cache_dir / "ava_further_improvements.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_name": best['name'],
            "best_val_acc": float(best['val_acc']),
            "best_test_acc": float(best['test_acc']),
            "all_improvements": [
                {"name": r['name'], "val_acc": float(r['val_acc']), "test_acc": float(r['test_acc'])}
                for r in improvements
            ],
        }, f, indent=2)
    
    log.info(f"Results saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

