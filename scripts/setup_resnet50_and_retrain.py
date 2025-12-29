#!/usr/bin/env python3
"""Set up ResNet50 embeddings and retrain the AVA multiclass model."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("resnet50_setup")


def build_resnet50_embeddings(images_dir: str, output_path: str, max_images: int = None):
    """Build ResNet50 embeddings for AVA images."""
    log.info("="*80)
    log.info("BUILDING RESNET50 EMBEDDINGS")
    log.info("="*80)
    
    if not os.path.exists(images_dir):
        log.error(f"Images directory not found: {images_dir}")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_images = len(image_files)
    log.info(f"Found {n_images} images in {images_dir}")
    
    if max_images:
        log.info(f"Limiting to {max_images} images")
    
    # Build embeddings
    cmd = [
        sys.executable, "scripts/build_cnn_embeddings.py",
        images_dir,
        "--model", "resnet50",
        "--batch-size", "16",  # Smaller batch for ResNet50 (larger model)
        "--output", output_path,
    ]
    
    if max_images:
        cmd.extend(["--limit", str(max_images)])
    
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        log.error(f"Failed to build embeddings:\n{result.stderr}")
        return False
    
    log.info("ResNet50 embeddings built successfully")
    return True


def retrain_with_resnet50(cache_dir: Path, max_ava: int = None, pca_dim: int = 192):
    """Retrain model with ResNet50 embeddings."""
    log.info("\n" + "="*80)
    log.info("RETRAINING WITH RESNET50 EMBEDDINGS")
    log.info("="*80)
    
    # Import here to avoid issues if dependencies not available
    import joblib
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.feature_transformer import FeatureInteractionTransformer
    
    # Load data
    ava_features_path = cache_dir / "ava_features.joblib"
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    ava_ids = ava_data.get('image_ids', [])
    
    if max_ava and len(X_ava) > max_ava:
        X_ava = X_ava[:max_ava]
        ava_ids = ava_ids[:max_ava]
    
    # Load labels
    ava_metadata_path = cache_dir / "ava_dataset" / "ava_downloader" / "AVA_dataset" / "AVA.txt"
    if not ava_metadata_path.exists():
        ava_metadata_path = cache_dir / "ava_dataset" / "AVA.txt"
    
    image_scores = {}
    if ava_metadata_path.exists():
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
    
    # Load ResNet50 embeddings
    embeddings_path = cache_dir / "embeddings_resnet50.joblib"
    if not embeddings_path.exists():
        log.error(f"ResNet50 embeddings not found at {embeddings_path}")
        log.error("Run: poetry run python scripts/setup_resnet50_and_retrain.py --build-embeddings")
        return False
    
    log.info(f"Loading ResNet50 embeddings from {embeddings_path}")
    emb_data = joblib.load(embeddings_path)
    emb = emb_data['embeddings']
    emb_fnames = emb_data.get('filenames', [])
    
    log.info(f"ResNet50 embeddings shape: {emb.shape} (original: 2048 dims)")
    
    # Align embeddings with features
    def align_and_concat(X_feats, filenames, emb, emb_fnames):
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
    
    filenames = [f"ava_{i}" for i in range(len(y_ava))]
    X_combined = align_and_concat(X_ava, filenames, emb, emb_fnames)
    
    log.info(f"Combined features: {X_combined.shape[1]} (78 handcrafted + {emb.shape[1]} ResNet50)")
    
    # Apply PCA
    if pca_dim and pca_dim < emb.shape[1]:
        log.info(f"Applying PCA: {emb.shape[1]} → {pca_dim} dimensions")
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_pca = pca.fit_transform(emb)
        X_combined = align_and_concat(X_ava, filenames, emb_pca, emb_fnames)
        log.info(f"After PCA: {X_combined.shape[1]} features")
    
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
    
    log.info(f"Final features: {X_final.shape[1]} (with interactions)")
    
    # Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_final, y_ava, test_size=0.2, stratify=y_ava, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1/0.8, stratify=y_trainval, random_state=42
    )
    
    log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train with best params
    from catboost import CatBoostClassifier
    
    cb_params = {
        "iterations": 2500,
        "learning_rate": 0.018,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "rsm": 0.88,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "random_seed": 42,
        "verbose": 50,
        "thread_count": -1,
        "loss_function": "MultiClass",
        "classes_count": 10,
        "early_stopping_rounds": 200,
        "eval_metric": "Accuracy",
        "use_best_model": True,
    }
    
    log.info("\nTraining CatBoost model with ResNet50 embeddings...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    
    # Evaluate
    cat_model = clf.named_steps['cat']
    X_val_scaled = clf.named_steps['scaler'].transform(X_val)
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    
    y_val_pred = cat_model.predict(X_val_scaled).flatten().astype(int)
    y_test_pred = cat_model.predict(X_test_scaled).flatten().astype(int)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    log.info(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    log.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Compare to baseline
    baseline_val = 0.4888
    improvement = val_acc - baseline_val
    log.info(f"\nBaseline (ResNet18): 48.88%")
    log.info(f"ResNet50: {val_acc*100:.2f}%")
    log.info(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        log.info("\n✅ ResNet50 embeddings IMPROVED the model!")
    else:
        log.info("\n⚠️  ResNet50 embeddings didn't improve (may need tuning)")
    
    # Classification report
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    target_names = [f"Score {i+1}" for i in unique_classes]
    log.info("\nClassification Report:")
    log.info(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=target_names, zero_division=0))
    
    # Save model
    model_path = cache_dir / "catboost_ava_multiclass_resnet50.joblib"
    joblib.dump(clf, model_path)
    log.info(f"\nModel saved to: {model_path}")
    
    # Save PCA if used
    if pca_dim and pca_dim < emb.shape[1]:
        pca_path = cache_dir / "pca_resnet50.joblib"
        joblib.dump(pca, pca_path)
        log.info(f"PCA transformer saved to: {pca_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Set up ResNet50 embeddings and retrain")
    parser.add_argument("--build-embeddings", action="store_true", help="Build ResNet50 embeddings")
    parser.add_argument("--retrain", action="store_true", help="Retrain model with ResNet50")
    parser.add_argument("--images-dir", default=".cache/ava_dataset/images", help="AVA images directory")
    parser.add_argument("--embeddings-output", default=".cache/embeddings_resnet50.joblib", help="Output path for embeddings")
    parser.add_argument("--max-ava", type=int, default=None, help="Max AVA samples")
    parser.add_argument("--pca-dim", type=int, default=192, help="PCA dimension for embeddings")
    parser.add_argument("--max-images", type=int, default=None, help="Max images for embedding extraction")
    args = parser.parse_args()
    
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Build embeddings if requested
    if args.build_embeddings:
        images_dir = os.path.expanduser(args.images_dir)
        output_path = os.path.expanduser(args.embeddings_output)
        if not build_resnet50_embeddings(images_dir, output_path, args.max_images):
            return 1
    
    # Retrain if requested
    if args.retrain:
        if not retrain_with_resnet50(cache_dir, args.max_ava, args.pca_dim):
            return 1
    
    # If neither specified, do both
    if not args.build_embeddings and not args.retrain:
        log.info("Building embeddings and retraining...")
        images_dir = os.path.expanduser(args.images_dir)
        output_path = os.path.expanduser(args.embeddings_output)
        if not build_resnet50_embeddings(images_dir, output_path, args.max_images):
            return 1
        if not retrain_with_resnet50(cache_dir, args.max_ava, args.pca_dim):
            return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

