#!/usr/bin/env python3
"""Train multi-class model on AVA dataset (0-10 scores) using tuned hyperparameters.

Usage:
    poetry run python scripts/train_ava_multiclass.py [--max-ava N] [--embeddings PATH]
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("train_ava_multiclass")


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
    # Try to load from metadata
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        # Try alternative location
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    if not ava_metadata_path.exists():
        log.error(f"AVA metadata not found at {ava_metadata_path}")
        return None, None
    
    log.info(f"Loading AVA metadata from {ava_metadata_path}...")
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
            
            # Use most frequent score as class label (1-10)
            most_frequent_score = np.argmax(score_counts) + 1  # 1-10
            # Alternative: use weighted mean rounded to nearest integer
            weighted_sum = sum((i+1) * count for i, count in enumerate(score_counts))
            mean_score = weighted_sum / total_votes
            rounded_score = int(round(mean_score))
            rounded_score = max(1, min(10, rounded_score))  # Clamp to 1-10
            
            image_scores[image_id] = {
                'class': rounded_score - 1,  # 0-9 for sklearn
                'mean_score': mean_score,
                'most_frequent': most_frequent_score - 1,  # 0-9
                'score_counts': score_counts,
            }
    
    log.info(f"Loaded scores for {len(image_scores)} images")
    return image_scores, ava_metadata_path


def main():
    parser = argparse.ArgumentParser(description="Train multi-class model on AVA dataset")
    parser.add_argument("--max-ava", type=int, default=None, help="Max AVA samples")
    parser.add_argument("--embeddings", type=str, default=None, help="Embeddings path")
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimensions")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set size")
    args = parser.parse_args()
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return 1
    
    log.info("="*80)
    log.info("MULTI-CLASS TRAINING ON AVA DATASET (0-10 SCORES)")
    log.info("="*80)
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Load tuned hyperparameters
    tuning_results_path = cache_dir / "catboost_ava_tuning_results.json"
    if tuning_results_path.exists():
        with open(tuning_results_path) as f:
            tuning_results = json.load(f)
        best_params = tuning_results['best_params']
        log.info(f"Using tuned hyperparameters from {tuning_results_path}")
        log.info(f"  Best CV accuracy: {tuning_results['best_cv_accuracy']:.4f}")
    else:
        log.warning("No tuning results found, using defaults")
        best_params = {}
    
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
    image_scores, metadata_path = load_ava_multiclass_labels(cache_dir)
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
        log.error("No matching labels found. Trying direct index mapping...")
        # Fallback: assume sequential mapping
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
                    y_ava.append(rounded - 1)  # 0-9
                else:
                    y_ava.append(4)  # Default to middle class
            valid_indices = list(range(len(X_ava)))
        else:
            log.error("No labels found")
            return 1
    
    X_ava = X_ava[valid_indices]
    y_ava = np.array(y_ava)
    
    log.info(f"Matched {len(y_ava)} samples with labels")
    log.info(f"Class distribution:")
    for cls in range(10):
        count = np.sum(y_ava == cls)
        if count > 0:
            log.info(f"  Class {cls+1}: {count} ({count/len(y_ava)*100:.1f}%)")
    
    # Load embeddings
    embeddings_path = args.embeddings
    if not embeddings_path:
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
        
        # Apply PCA
        if args.pca_dim and args.pca_dim < emb.shape[1]:
            log.info(f"Applying PCA to reduce embeddings to {args.pca_dim} dimensions...")
            pca = PCA(n_components=args.pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
            log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Concatenate features and embeddings
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
    
    # Log class distribution
    present_classes = np.unique(y_train)
    class_counts = {cls: int(np.sum(y_train == cls)) for cls in present_classes}
    
    log.info(f"\nClass distribution (train):")
    for cls in sorted(present_classes):
        count = class_counts[cls]
        log.info(f"  Class {cls} (Score {cls+1}): {count:5d} ({count/len(y_train)*100:5.1f}%)")
    
    # Build model - optimized configuration (achieved 50.91% validation accuracy)
    cb_params = {
        "iterations": best_params.get("iterations", 1000) * 2,  # 2000 iterations
        "learning_rate": best_params.get("learning_rate", 0.038) * 0.6,  # 0.023
        "depth": best_params.get("depth", 8) - 1,  # 7
        "l2_leaf_reg": best_params.get("l2_leaf_reg", 3.5) * 1.5,  # 5.3
        "rsm": best_params.get("rsm", 0.85) * 0.95,  # 0.81
        "bootstrap_type": "Bernoulli",  # Required for subsample
        "subsample": best_params.get("subsample", 0.83) * 0.95,  # 0.79
        "random_seed": 42,
        "verbose": 50,
        "thread_count": -1,
        "loss_function": "MultiClass",  # Multi-class classification
        "classes_count": 10,  # 10 classes (0-9, representing 1-10 scores)
        "early_stopping_rounds": 200,  # Early stopping patience
        "eval_metric": "Accuracy",  # Use accuracy for early stopping
        "use_best_model": True,  # Use best model from validation
    }
    
    log.info(f"\nTraining configuration:")
    for key, value in cb_params.items():
        log.info(f"  {key}: {value}")
    
    # Create pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    log.info("\n" + "="*80)
    log.info("TRAINING MULTI-CLASS MODEL")
    log.info("="*80)
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    # Evaluate
    log.info("\n" + "="*80)
    log.info("EVALUATION RESULTS")
    log.info("="*80)
    
    # Get predictions from the CatBoost model directly (not pipeline)
    cat_model = clf.named_steps['cat']
    X_train_scaled = clf.named_steps['scaler'].transform(X_train)
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_train_pred = cat_model.predict(X_train_scaled)
    y_val_pred = cat_model.predict(X_val_scaled)
    y_test_pred = cat_model.predict(X_test_scaled)
    
    # Ensure predictions are 1D integers
    y_train_pred = y_train_pred.flatten().astype(int)
    y_val_pred = y_val_pred.flatten().astype(int)
    y_test_pred = y_test_pred.flatten().astype(int)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    log.info(f"\nAccuracy:")
    log.info(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    log.info(f"  Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    log.info(f"  Test: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    overfitting = train_acc - val_acc
    log.info(f"\nOverfitting gap: {overfitting:.4f} ({overfitting*100:.2f}%)")
    
    # Get unique classes that actually exist
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    target_names = [f"Score {i+1}" for i in unique_classes]
    
    log.info("\nClassification Report (Test Set):")
    log.info(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names))
    
    log.info("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred, labels=unique_classes)
    log.info(f"\n{cm}")
    
    # Save model
    model_path = cache_dir / "catboost_ava_multiclass.joblib"
    joblib.dump(clf, model_path)
    log.info(f"\nModel saved to: {model_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

