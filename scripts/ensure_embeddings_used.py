#!/usr/bin/env python3
"""Ensure embeddings are directly used by forcing embedding-handcrafted interactions."""

from __future__ import annotations

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

from src.feature_transformer import FeatureInteractionTransformer

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

def prepare_data_with_forced_emb_interactions(cache_dir: Path, max_ava: int = None, pca_dim: int = 128):
    """Prepare data with forced embedding-handcrafted interactions."""
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
            import json
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
    
    X = X_ava
    y = y_ava
    filenames = [f"ava_{i}" for i in range(len(y_ava))]
    
    # Load embeddings
    embeddings_path = None
    for p in [cache_dir / "embeddings_resnet18_full.joblib", cache_dir / "embeddings_resnet18.joblib"]:
        if p.exists():
            embeddings_path = str(p)
            break
    
    if embeddings_path:
        emb, emb_fnames = load_embeddings(embeddings_path)
        if pca_dim and pca_dim < emb.shape[1]:
            pca = PCA(n_components=pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
        X_combined = align_and_concat(X, filenames, emb, emb_fnames)
    else:
        X_combined = X
    
    n_handcrafted = X.shape[1]  # 78
    n_embeddings = emb.shape[1] if embeddings_path else 0  # 128
    
    # Standard interactions (top features by variance)
    top_n = 15
    feature_vars = np.var(X_combined, axis=0)
    top_indices = np.argsort(feature_vars)[::-1][:top_n]
    
    interaction_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < 80:  # Reserve space for forced interactions
                interaction_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    # FORCE embedding-handcrafted interactions
    # Top handcrafted features
    handcrafted_vars = feature_vars[:n_handcrafted]
    top_handcrafted = np.argsort(handcrafted_vars)[::-1][:10]  # Top 10 handcrafted
    
    # Top embedding features
    if n_embeddings > 0:
        embedding_vars = feature_vars[n_handcrafted:n_handcrafted+n_embeddings]
        top_embeddings = np.argsort(embedding_vars)[::-1][:10]  # Top 10 embeddings
        top_embeddings_abs = [n_handcrafted + idx for idx in top_embeddings]
        
        # Create forced interactions: top handcrafted * top embeddings
        for hc_idx in top_handcrafted:
            for emb_idx in top_embeddings_abs:
                if len(interaction_pairs) < 100:
                    interaction_pairs.append((int(hc_idx), int(emb_idx)))
    
    # Ratios
    ratio_pairs = []
    for i in range(min(20, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 5, len(top_indices))):
            if len(ratio_pairs) < 20:
                ratio_pairs.append((int(top_indices[i]), int(top_indices[j])))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    return X_final, y, len(interaction_pairs)

def train_and_evaluate(name: str, X_train, X_val, X_test, y_train, y_val, y_test):
    from catboost import CatBoostClassifier
    
    params = {
        "iterations": 2500,
        "learning_rate": 0.018,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "rsm": 0.88,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
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
        ("cat", CatBoostClassifier(**params)),
    ])
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    cat_model = clf.named_steps['cat']
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_val_pred = cat_model.predict(X_val_scaled).flatten().astype(int)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    return {"name": name, "val_acc": val_acc, "test_acc": test_acc, "model": clf}

def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    print("="*80)
    print("ENSURING EMBEDDINGS ARE DIRECTLY USED")
    print("="*80)
    
    # Prepare data with forced embedding interactions
    print("\nPreparing data with FORCED embedding-handcrafted interactions...")
    data = prepare_data_with_forced_emb_interactions(cache_dir, max_ava=10000, pca_dim=128)
    if data is None:
        print("Failed to prepare data")
        return 1
    
    X_final, y, n_interactions = data
    print(f"  Total features: {X_final.shape[1]}")
    print(f"  Interactions: {n_interactions} (including forced embedding-handcrafted)")
    
    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    print("\nTraining model with forced embedding interactions...")
    result = train_and_evaluate("forced_emb_interactions", X_train, X_val, X_test, y_train, y_val, y_test)
    
    if result:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Configuration: {result['name']}")
        print(f"Validation Accuracy: {result['val_acc']:.4f} ({result['val_acc']*100:.2f}%)")
        print(f"Test Accuracy: {result['test_acc']:.4f} ({result['test_acc']*100:.2f}%)")
        
        # Compare to baseline
        baseline_val = 0.4888  # From previous best
        improvement = result['val_acc'] - baseline_val
        print(f"\nBaseline (standard interactions): 48.88%")
        print(f"With forced embedding interactions: {result['val_acc']*100:.2f}%")
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if improvement > 0:
            print("\n✅ Forced embedding interactions HELP!")
        else:
            print("\n⚠️  Forced embedding interactions didn't improve (embeddings already used via variance selection)")
        
        # Save model
        model_path = cache_dir / "catboost_ava_with_forced_embeddings.joblib"
        joblib.dump(result['model'], model_path)
        print(f"\nModel saved to: {model_path}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

