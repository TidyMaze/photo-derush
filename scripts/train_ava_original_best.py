#!/usr/bin/env python3
"""Reproduce and improve the original 52.03% validation accuracy model."""

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
log = logging.getLogger("train_ava_original")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ava", type=int, default=None)
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("ORIGINAL BEST CONFIGURATION (52.03% target)")
    log.info("="*80)
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Load data
    ava_features_path = cache_dir / "ava_features.joblib"
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
    if args.max_ava and len(X_ava) > args.max_ava:
        X_ava = X_ava[:args.max_ava]
        ava_ids = ava_ids[:args.max_ava]
    
    image_scores = load_ava_multiclass_labels(cache_dir)
    if image_scores is None:
        return 1
    
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
    
    # Load embeddings
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
        pca = PCA(n_components=128, random_state=42)
        emb = pca.fit_transform(emb)
        X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    else:
        X_combined = X
    
    # Feature interactions
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
    
    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
    )
    
    log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Original successful config (52.03%)
    from catboost import CatBoostClassifier
    
    # Try original exact config
    configs = [
        ("original_exact", {
            "iterations": 1000,
            "learning_rate": 0.019,  # 0.038 * 0.5
            "depth": 7,  # 8 - 1
            "l2_leaf_reg": 7.0,  # 3.5 * 2
            "rsm": 0.765,  # 0.85 * 0.9
            "bootstrap_type": "Bernoulli",
            "subsample": 0.747,  # 0.83 * 0.9
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "loss_function": "MultiClass",
            "classes_count": 10,
            "early_stopping_rounds": 50,
            "eval_metric": "Accuracy",
            "use_best_model": False,  # Original didn't use this
        }),
        ("original_with_best_model", {
            "iterations": 1000,
            "learning_rate": 0.019,
            "depth": 7,
            "l2_leaf_reg": 7.0,
            "rsm": 0.765,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.747,
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "loss_function": "MultiClass",
            "classes_count": 10,
            "early_stopping_rounds": 50,
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }),
        ("original_extended", {
            "iterations": 1500,
            "learning_rate": 0.019,
            "depth": 7,
            "l2_leaf_reg": 7.0,
            "rsm": 0.765,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.747,
            "random_seed": 42,
            "verbose": 50,
            "thread_count": -1,
            "loss_function": "MultiClass",
            "classes_count": 10,
            "early_stopping_rounds": 100,
            "eval_metric": "Accuracy",
            "use_best_model": True,
        }),
    ]
    
    best_result = None
    best_val_acc = 0
    
    for config_name, cb_params in configs:
        log.info(f"\nTrying {config_name}...")
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
        
        log.info(f"  Val: {val_acc:.4f} ({val_acc*100:.2f}%), Test: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_result = {
                "config": config_name,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "model": clf,
            }
    
    if best_result:
        log.info("\n" + "="*80)
        log.info("BEST FINAL MODEL")
        log.info("="*80)
        log.info(f"Configuration: {best_result['config']}")
        log.info(f"Validation accuracy: {best_result['val_acc']:.4f} ({best_result['val_acc']*100:.2f}%)")
        log.info(f"Test accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc']*100:.2f}%)")
        
        # Final evaluation
        model = best_result['model']
        cat_model = model.named_steps['cat']
        X_test_scaled = model.named_steps['scaler'].transform(X_test)
        y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
        
        unique_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
        target_names = [f"Score {i+1}" for i in unique_classes]
        
        log.info("\nFinal Classification Report:")
        log.info(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names, zero_division=0))
        
        # Save
        model_path = cache_dir / "catboost_ava_multiclass_ultimate_best.joblib"
        joblib.dump(model, model_path)
        log.info(f"\nUltimate best model saved to: {model_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

