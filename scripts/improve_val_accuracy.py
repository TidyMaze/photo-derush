#!/usr/bin/env python3
"""Test multiple strategies to improve validation accuracy.

Usage:
    poetry run python scripts/improve_val_accuracy.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("improve_val")


def optimize_threshold(y_true, y_proba):
    """Find optimal threshold that maximizes accuracy."""
    thresholds = np.arange(0.3, 0.71, 0.01)
    best_threshold = 0.5
    best_accuracy = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy


def test_configuration(X_train, X_val, y_train, y_val, config_name, cb_params):
    """Test a specific configuration and return results."""
    log.info(f"\n{'='*80}")
    log.info(f"Testing: {config_name}")
    log.info(f"{'='*80}")
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params_final = {
        **cb_params,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
        "use_best_model": False,
    }
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params_final)),
    ])
    
    clf.fit(X_train, y_train)
    
    # Get probabilities
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    
    # Optimize threshold
    best_threshold, best_val_acc = optimize_threshold(y_val, y_val_proba)
    
    # Default threshold (0.5)
    y_val_pred_default = (y_val_proba >= 0.5).astype(int)
    default_val_acc = accuracy_score(y_val, y_val_pred_default)
    
    log.info(f"  Default threshold (0.5): {default_val_acc:.4f} ({default_val_acc*100:.2f}%)")
    log.info(f"  Optimal threshold ({best_threshold:.3f}): {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    log.info(f"  Improvement: +{(best_val_acc - default_val_acc)*100:.2f} pp")
    
    return {
        "config_name": config_name,
        "default_val_acc": float(default_val_acc),
        "optimal_val_acc": float(best_val_acc),
        "optimal_threshold": float(best_threshold),
        "improvement": float(best_val_acc - default_val_acc),
        "params": cb_params,
    }


def main():
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    
    # Load AVA features
    ava_features_path = cache_dir / "ava_features.joblib"
    if not ava_features_path.exists():
        log.error(f"AVA features not found at {ava_features_path}")
        return 1
    
    log.info("Loading AVA features...")
    ava_data = joblib.load(ava_features_path)
    X_ava = ava_data['features']
    y_ava = ava_data['labels']
    
    log.info(f"AVA dataset: {len(y_ava)} samples")
    
    # Load embeddings and apply PCA (same as train_with_ava.py)
    log.info("Loading embeddings...")
    embeddings_path = cache_dir / "embeddings_resnet18.joblib"
    if embeddings_path.exists():
        embeddings_data = joblib.load(embeddings_path)
        all_embeddings = embeddings_data['embeddings']
        all_filenames = embeddings_data.get('filenames', [])
        
        # Get AVA image IDs
        ava_labels_path = cache_dir / "ava_dataset" / "ava_keep_trash_labels.json"
        if ava_labels_path.exists():
            import json
            with open(ava_labels_path) as f:
                ava_labeled = json.load(f)
            ava_image_ids = {str(item['image_id']) for item in ava_labeled}
            
            # Align embeddings with AVA features
            aligned_embeddings = []
            for i, ava_id in enumerate(range(len(X_ava))):
                # Try to find matching embedding
                # AVA images are named like "145.jpg", so look for "145" in filenames
                ava_id_str = str(ava_id)
                matching_idx = None
                for j, fn in enumerate(all_filenames):
                    if ava_id_str in fn or fn.replace('.jpg', '') == ava_id_str:
                        matching_idx = j
                        break
                
                if matching_idx is not None:
                    aligned_embeddings.append(all_embeddings[matching_idx])
                else:
                    # Use zero embedding if not found
                    aligned_embeddings.append(np.zeros(all_embeddings.shape[1]))
            
            embeddings = np.array(aligned_embeddings)
        else:
            # Fallback: use first N embeddings
            embeddings = all_embeddings[:len(X_ava)]
        
        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=128)
        embeddings_pca = pca.fit_transform(embeddings)
        log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Combine features
        X_combined = np.hstack([X_ava, embeddings_pca])
    else:
        X_combined = X_ava
    
    # Add feature interactions
    from src.feature_transformer import FeatureInteractionTransformer
    
    # Load interaction pairs (from previous training)
    interaction_pairs = []
    ratio_pairs = []
    # Use top 15 features for interactions
    top_n = 15
    feature_importances = np.ones(X_combined.shape[1])  # Placeholder
    top_indices = np.argsort(feature_importances)[-top_n:]
    
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < 100:
                interaction_pairs.append((top_indices[i], top_indices[j]))
    
    for i in range(min(20, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 5, len(top_indices))):
            if len(ratio_pairs) < 20:
                ratio_pairs.append((top_indices[i], top_indices[j]))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    log.info(f"Final feature count: {X_final.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_ava, test_size=0.2, stratify=y_ava, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")
    
    # Test different configurations
    results = []
    
    # Baseline (current best)
    results.append(test_configuration(
        X_train, X_val, y_train, y_val,
        "Baseline (iterations=2000, depth=7, l2=3.0, lr=0.06)",
        {
            "iterations": 2000,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "learning_rate": 0.06,
        }
    ))
    
    # More regularization
    results.append(test_configuration(
        X_train, X_val, y_train, y_val,
        "More regularization (iterations=2000, depth=6, l2=5.0, lr=0.05)",
        {
            "iterations": 2000,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "learning_rate": 0.05,
        }
    ))
    
    # Even more regularization
    results.append(test_configuration(
        X_train, X_val, y_train, y_val,
        "Strong regularization (iterations=2000, depth=5, l2=8.0, lr=0.04)",
        {
            "iterations": 2000,
            "depth": 5,
            "l2_leaf_reg": 8.0,
            "learning_rate": 0.04,
        }
    ))
    
    # Lower learning rate, more iterations
    results.append(test_configuration(
        X_train, X_val, y_train, y_val,
        "Lower LR (iterations=3000, depth=7, l2=3.0, lr=0.04)",
        {
            "iterations": 3000,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "learning_rate": 0.04,
        }
    ))
    
    # Balanced approach
    results.append(test_configuration(
        X_train, X_val, y_train, y_val,
        "Balanced (iterations=2500, depth=6, l2=4.0, lr=0.05)",
        {
            "iterations": 2500,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "learning_rate": 0.05,
        }
    ))
    
    # Find best configuration
    best_result = max(results, key=lambda x: x["optimal_val_acc"])
    
    log.info("\n" + "="*80)
    log.info("SUMMARY")
    log.info("="*80)
    for r in sorted(results, key=lambda x: x["optimal_val_acc"], reverse=True):
        log.info(f"{r['config_name']}: {r['optimal_val_acc']:.4f} ({r['optimal_val_acc']*100:.2f}%)")
    
    log.info(f"\nBest configuration: {best_result['config_name']}")
    log.info(f"Validation accuracy: {best_result['optimal_val_acc']:.4f} ({best_result['optimal_val_acc']*100:.2f}%)")
    log.info(f"Optimal threshold: {best_result['optimal_threshold']:.3f}")
    
    # Save results
    results_path = cache_dir / "val_accuracy_improvement_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

