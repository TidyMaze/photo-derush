#!/usr/bin/env python3
"""Verify that embeddings are actually used and helping the model."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
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

def prepare_data(cache_dir: Path, include_embeddings: bool = True, pca_dim: int = 128):
    """Prepare data with or without embeddings."""
    ava_features_path = cache_dir / "ava_features.joblib"
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
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
    
    if include_embeddings:
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
    else:
        X_combined = X
    
    # Add feature interactions
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
    
    return X_final, y, include_embeddings

def train_and_evaluate(name: str, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train model and return results."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    
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
    
    # Get feature importances
    importances = cat_model.get_feature_importance()
    n_features = len(importances)
    
    return {
        "name": name,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "n_features": n_features,
        "importances": importances,
        "model": clf,
    }

def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    print("="*80)
    print("VERIFYING EMBEDDINGS IMPACT")
    print("="*80)
    
    # Prepare data with embeddings
    print("\nPreparing data WITH embeddings...")
    data_with = prepare_data(cache_dir, include_embeddings=True, pca_dim=128)
    if data_with is None:
        print("Failed to prepare data")
        return 1
    
    X_with, y_with, _ = data_with
    print(f"  Features with embeddings: {X_with.shape[1]}")
    
    # Prepare data without embeddings
    print("\nPreparing data WITHOUT embeddings...")
    data_without = prepare_data(cache_dir, include_embeddings=False)
    if data_without is None:
        print("Failed to prepare data")
        return 1
    
    X_without, y_without, _ = data_without
    print(f"  Features without embeddings: {X_without.shape[1]}")
    
    # Split data (same splits for fair comparison)
    X_trainval_with, X_test_with, y_trainval, y_test = train_test_split(
        X_with, y_with, test_size=0.2, stratify=y_with, random_state=42
    )
    X_train_with, X_val_with, y_train, y_val = train_test_split(
        X_trainval_with, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
    )
    
    # Get same indices for without embeddings
    X_train_without = X_without[:len(X_train_with)]
    X_val_without = X_without[len(X_train_with):len(X_train_with)+len(X_val_with)]
    X_test_without = X_without[len(X_train_with)+len(X_val_with):]
    
    print(f"\nTrain: {len(X_train_with)}, Val: {len(X_val_with)}, Test: {len(X_test_with)}")
    
    # Train with embeddings
    print("\n" + "="*80)
    print("TRAINING WITH EMBEDDINGS")
    print("="*80)
    result_with = train_and_evaluate("with_embeddings", X_train_with, X_val_with, X_test_with, 
                                     y_train, y_val, y_test)
    
    # Train without embeddings
    print("\n" + "="*80)
    print("TRAINING WITHOUT EMBEDDINGS")
    print("="*80)
    result_without = train_and_evaluate("without_embeddings", X_train_without, X_val_without, X_test_without,
                                        y_train, y_val, y_test)
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if result_with and result_without:
        print(f"\nWITH EMBEDDINGS:")
        print(f"  Features: {result_with['n_features']}")
        print(f"  Validation Accuracy: {result_with['val_acc']:.4f} ({result_with['val_acc']*100:.2f}%)")
        print(f"  Test Accuracy: {result_with['test_acc']:.4f} ({result_with['test_acc']*100:.2f}%)")
        
        print(f"\nWITHOUT EMBEDDINGS:")
        print(f"  Features: {result_without['n_features']}")
        print(f"  Validation Accuracy: {result_without['val_acc']:.4f} ({result_without['val_acc']*100:.2f}%)")
        print(f"  Test Accuracy: {result_without['test_acc']:.4f} ({result_without['test_acc']*100:.2f}%)")
        
        val_diff = result_with['val_acc'] - result_without['val_acc']
        test_diff = result_with['test_acc'] - result_without['test_acc']
        
        print(f"\nIMPROVEMENT FROM EMBEDDINGS:")
        print(f"  Validation: {val_diff:+.4f} ({val_diff*100:+.2f}%)")
        print(f"  Test: {test_diff:+.4f} ({test_diff*100:+.2f}%)")
        
        if val_diff > 0:
            print(f"\n✅ Embeddings HELP - {val_diff*100:.2f}% improvement")
        elif val_diff < -0.01:
            print(f"\n❌ Embeddings HURT - {abs(val_diff)*100:.2f}% worse")
        else:
            print(f"\n⚠️  Embeddings NEUTRAL - no significant difference")
        
        # Analyze embedding feature importances
        print(f"\n" + "="*80)
        print("EMBEDDING FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        n_handcrafted = 78
        n_embeddings = 128
        n_base = n_handcrafted + n_embeddings
        
        # Get importances for embedding features
        emb_importances = result_with['importances'][n_handcrafted:n_base]
        handcrafted_importances = result_with['importances'][:n_handcrafted]
        
        print(f"\nHandcrafted features (0-{n_handcrafted-1}):")
        print(f"  Mean importance: {np.mean(handcrafted_importances):.4f}")
        print(f"  Max importance: {np.max(handcrafted_importances):.4f}")
        print(f"  Sum importance: {np.sum(handcrafted_importances):.4f}")
        
        print(f"\nEmbedding features ({n_handcrafted}-{n_base-1}):")
        print(f"  Mean importance: {np.mean(emb_importances):.4f}")
        print(f"  Max importance: {np.max(emb_importances):.4f}")
        print(f"  Sum importance: {np.sum(emb_importances):.4f}")
        
        # Top embedding features
        top_emb_indices = np.argsort(emb_importances)[::-1][:10]
        print(f"\nTop 10 embedding features:")
        for rank, idx in enumerate(top_emb_indices, 1):
            orig_idx = n_handcrafted + idx
            print(f"  {rank:2d}. embedding_pca_{idx:3d} (index {orig_idx:3d}): {emb_importances[idx]:.4f}")
        
        # Check if any embeddings are in top features overall
        all_top_indices = np.argsort(result_with['importances'])[::-1][:20]
        emb_in_top = [idx for idx in all_top_indices if n_handcrafted <= idx < n_base]
        print(f"\nEmbeddings in top 20 features: {len(emb_in_top)}")
        if emb_in_top:
            print(f"  Indices: {[idx - n_handcrafted for idx in emb_in_top]}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

