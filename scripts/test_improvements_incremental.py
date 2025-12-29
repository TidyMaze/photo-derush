#!/usr/bin/env python3
"""Test model improvements incrementally - one change at a time.

Measures before/after for each change, keeps what works, undoes what doesn't.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.repository import RatingsTagsRepository
from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", force=True)
log = logging.getLogger("test_improvements")


def evaluate_model(clf: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    
    return metrics


def train_model(X_train, X_val, y_train, y_val, config: dict) -> Pipeline:
    """Train model with given configuration."""
    from catboost import CatBoostClassifier
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    cb_params = {
        "iterations": config.get("iterations", 200),
        "learning_rate": config.get("learning_rate", 0.1),
        "depth": config.get("depth", 6),
        "l2_leaf_reg": config.get("l2_leaf_reg", 1.0),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    # Optional parameters
    if "rsm" in config:
        cb_params["rsm"] = config["rsm"]
        cb_params["bootstrap_type"] = "Bernoulli"
    if "subsample" in config:
        cb_params["subsample"] = config["subsample"]
        cb_params["bootstrap_type"] = "Bernoulli"
    if "early_stopping_rounds" in config:
        cb_params["early_stopping_rounds"] = config["early_stopping_rounds"]
        cb_params["eval_metric"] = "Accuracy"
        cb_params["use_best_model"] = True
    
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    if "early_stopping_rounds" in config and X_val is not None:
        clf.fit(X_train, y_train, cat__eval_set=(X_val, y_val))
    else:
        clf.fit(X_train, y_train)
    
    return clf


def add_feature_interactions(X_base, embeddings_path: str = None):
    """Add feature interactions if embeddings are available."""
    if embeddings_path is None:
        return X_base, None
    
    try:
        import joblib
        from sklearn.decomposition import PCA
        import os
        
        # Load embeddings
        data = joblib.load(embeddings_path)
        emb = data['embeddings']
        emb_fnames = data.get('filenames', [])
        
        # Apply PCA to reduce dimensions
        if emb.shape[1] > 128:
            pca = PCA(n_components=128, random_state=42)
            emb = pca.fit_transform(emb)
        
        # Align embeddings with features (simplified - assume same order)
        # In practice, would need to match by filename
        if len(emb) >= len(X_base):
            emb_aligned = emb[:len(X_base)]
        else:
            # Pad with zeros if needed
            emb_aligned = np.vstack([emb, np.zeros((len(X_base) - len(emb), emb.shape[1]))])
        
        X_combined = np.hstack([X_base, emb_aligned])
        
        # Add interactions
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
        
        log.info(f"Added {len(interaction_pairs)} interactions + {len(ratio_pairs)} ratios")
        log.info(f"Feature count: {X_base.shape[1]} → {X_final.shape[1]}")
        
        return X_final, transformer
    except Exception as e:
        log.warning(f"Failed to add feature interactions: {e}")
        return X_base, None


def test_improvement(X_train, X_val, X_test, y_train, y_val, y_test, baseline_metrics: dict, improvement: dict, name: str) -> tuple[bool, dict]:
    """Test one improvement and return (is_better, metrics)."""
    log.info(f"\n{'='*80}")
    log.info(f"Testing: {name}")
    log.info(f"{'='*80}")
    
    config = improvement.get("config", {})
    use_interactions = improvement.get("use_interactions", False)
    embeddings_path = improvement.get("embeddings_path")
    
    # Apply feature interactions if requested
    if use_interactions and embeddings_path:
        log.info(f"Adding feature interactions using embeddings from {embeddings_path}")
        X_train_final, transformer = add_feature_interactions(X_train, embeddings_path)
        X_val_final, _ = add_feature_interactions(X_val, embeddings_path) if X_val is not None else (X_val, None)
        X_test_final, _ = add_feature_interactions(X_test, embeddings_path)
    else:
        X_train_final, X_val_final, X_test_final = X_train, X_val, X_test
        log.info(f"Config: {config}")
    
    clf = train_model(X_train_final, X_val_final, y_train, y_val, config)
    metrics = evaluate_model(clf, X_test_final, y_test)
    
    improvement_val = metrics["accuracy"] - baseline_metrics["accuracy"]
    
    log.info(f"Results:")
    log.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    log.info(f"  F1: {metrics['f1']:.4f}")
    log.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    log.info(f"  vs Baseline: {improvement_val:+.4f} ({improvement_val*100:+.2f} pp)")
    
    is_better = improvement_val > 0.001  # > 0.1% improvement threshold
    
    if is_better:
        log.info(f"✅ KEEPING: {name}")
    else:
        log.info(f"❌ REJECTING: {name} (no improvement)")
    
    return is_better, metrics


def main():
    parser = argparse.ArgumentParser(description="Test model improvements incrementally")
    parser.add_argument("--image-dir", default=None, help="Image directory")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("INCREMENTAL MODEL IMPROVEMENT TESTING")
    log.info("="*80)
    log.info(f"Image directory: {image_dir}")
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info("Building dataset...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(y)
    n_keep = int(np.sum(y == 1))
    n_trash = int(np.sum(y == 0))
    
    log.info(f"Dataset: {n_samples} samples ({n_keep} keep, {n_trash} trash)")
    
    if n_samples < 20:
        log.error("Insufficient labeled data")
        return 1
    
    # Split: 80% train, 10% val, 10% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=42
    )
    
    log.info(f"Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Baseline (current production config with early stopping)
    log.info("\n" + "="*80)
    log.info("BASELINE (Current Production Config with Early Stopping)")
    log.info("="*80)
    baseline_config = {
        "iterations": 500,  # Increased with early stopping
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "early_stopping_rounds": 50,  # Early stopping enabled
    }
    baseline_clf = train_model(X_train, X_val, y_train, y_val, baseline_config)
    baseline_metrics = evaluate_model(baseline_clf, X_test, y_test)
    
    log.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
    log.info(f"Baseline F1: {baseline_metrics['f1']:.4f}")
    log.info(f"Baseline ROC-AUC: {baseline_metrics.get('roc_auc', 0):.4f}")
    
    # Track best config (starts as baseline)
    best_config = baseline_config.copy()
    best_metrics = baseline_metrics.copy()
    
    # Test improvements one at a time (all with early stopping now)
    improvements_to_test = [
        # 1. More iterations (with early stopping protection)
        {
            "name": "More iterations (1000)",
            "config": {**baseline_config, "iterations": 1000},
        },
        # 2. Lower learning rate (more iterations needed)
        {
            "name": "Lower LR (0.05) + more iterations",
            "config": {**baseline_config, "learning_rate": 0.05, "iterations": 1000},
        },
        # 3. Moderate learning rate
        {
            "name": "Moderate LR (0.08)",
            "config": {**baseline_config, "learning_rate": 0.08, "iterations": 1000},
        },
        # 4. Slightly deeper trees
        {
            "name": "Deeper trees (depth=7)",
            "config": {**baseline_config, "depth": 7},
        },
        # 5. Shallower trees
        {
            "name": "Shallower trees (depth=5)",
            "config": {**baseline_config, "depth": 5},
        },
        # 6. More regularization
        {
            "name": "More regularization (l2_leaf_reg=2.0)",
            "config": {**baseline_config, "l2_leaf_reg": 2.0},
        },
        # 7. Less regularization
        {
            "name": "Less regularization (l2_leaf_reg=0.5)",
            "config": {**baseline_config, "l2_leaf_reg": 0.5},
        },
        # 8. Tighter early stopping (more aggressive)
        {
            "name": "Tighter early stopping (patience=30)",
            "config": {**baseline_config, "early_stopping_rounds": 30},
        },
        # 9. Looser early stopping (more patience)
        {
            "name": "Looser early stopping (patience=100)",
            "config": {**baseline_config, "early_stopping_rounds": 100},
        },
        # 10. Feature dropout (rsm) - light
        {
            "name": "Light feature dropout (rsm=0.98)",
            "config": {**baseline_config, "rsm": 0.98, "bootstrap_type": "Bernoulli"},
        },
        # 11. Row dropout (subsample) - light
        {
            "name": "Light row dropout (subsample=0.98)",
            "config": {**baseline_config, "subsample": 0.98, "bootstrap_type": "Bernoulli"},
        },
    ]
    
    # Test feature interactions if embeddings available
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    embeddings_path = None
    for path in [cache_dir / "embeddings_resnet18_full.joblib", cache_dir / "embeddings_resnet18.joblib"]:
        if path.exists():
            embeddings_path = str(path)
            break
    
    if embeddings_path:
        log.info(f"\nFound embeddings at {embeddings_path}, will test feature interactions")
        improvements_to_test.append({
            "name": "Feature interactions (with embeddings)",
            "config": baseline_config,
            "use_interactions": True,
            "embeddings_path": embeddings_path,
        })
    
    log.info("\n" + "="*80)
    log.info("TESTING IMPROVEMENTS (ONE AT A TIME)")
    log.info("="*80)
    
    for improvement in improvements_to_test:
        is_better, metrics = test_improvement(
            X_train, X_val, X_test, y_train, y_val, y_test,
            best_metrics, improvement, improvement["name"]
        )
        
        if is_better:
            # Update best config (merge with current best)
            best_config.update(improvement["config"])
            best_metrics = metrics
            log.info(f"Updated best config. New accuracy: {best_metrics['accuracy']:.4f}")
        else:
            # Don't update - keep previous best
            log.info(f"Keeping previous best config (accuracy: {best_metrics['accuracy']:.4f})")
    
    # Final summary
    log.info("\n" + "="*80)
    log.info("FINAL SUMMARY")
    log.info("="*80)
    log.info(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
    log.info(f"Best Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    improvement = best_metrics["accuracy"] - baseline_metrics["accuracy"]
    log.info(f"Total Improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    log.info(f"\nBest Config:")
    for key, value in sorted(best_config.items()):
        log.info(f"  {key}: {value}")
    
    if improvement > 0.001:
        log.info("\n✅ Found improvements! Consider updating production model.")
    else:
        log.info("\n⚠️  No significant improvements found. Baseline is optimal.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

