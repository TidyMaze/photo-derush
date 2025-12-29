#!/usr/bin/env python3
"""Tune CatBoost hyperparameters on AVA dataset using Optuna.

Usage:
    poetry run python scripts/tune_ava.py [--n-trials N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_transformer import FeatureInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("tune_ava")


def objective(trial, X, y, scale_pos_weight):
    """Optuna objective function for CatBoost hyperparameter tuning."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        raise
    
    # Suggest hyperparameters
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),  # Feature dropout
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),  # Row dropout
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    # Create pipeline
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**params)),
    ])
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    
    return scores.mean()


def main():
    parser = argparse.ArgumentParser(description="Tune CatBoost hyperparameters on AVA dataset")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    
    try:
        import optuna
    except ImportError:
        log.error("Optuna not available. Install with: poetry add optuna")
        return 1
    
    log.info("="*80)
    log.info("CATBOOST HYPERPARAMETER TUNING ON AVA DATASET")
    log.info("="*80)
    
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
    
    # Load embeddings and apply PCA (skip if not aligned)
    embeddings_path = cache_dir / "embeddings_resnet18.joblib"
    if embeddings_path.exists():
        embeddings_data = joblib.load(embeddings_path)
        all_embeddings = embeddings_data['embeddings']
        
        # Check if embeddings match AVA size
        if len(all_embeddings) == len(X_ava):
            log.info("Loading embeddings...")
            embeddings = all_embeddings
            
            # Apply PCA
            pca = PCA(n_components=128, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings)
            log.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            
            # Combine features
            X_combined = np.hstack([X_ava, embeddings_pca])
        else:
            log.info(f"Skipping embeddings: size mismatch ({len(all_embeddings)} vs {len(X_ava)})")
            X_combined = X_ava
    else:
        X_combined = X_ava
    
    # Add feature interactions
    log.info("Adding feature interactions...")
    top_n = 15
    feature_importances = np.ones(X_combined.shape[1])
    top_indices = np.argsort(feature_importances)[-top_n:]
    
    interaction_pairs = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            if len(interaction_pairs) < 100:
                interaction_pairs.append((top_indices[i], top_indices[j]))
    
    ratio_pairs = []
    for i in range(min(20, len(top_indices))):
        for j in range(i + 1, min(i + 1 + 5, len(top_indices))):
            if len(ratio_pairs) < 20:
                ratio_pairs.append((top_indices[i], top_indices[j]))
    
    transformer = FeatureInteractionTransformer(interaction_pairs, ratio_pairs)
    X_final = transformer.transform(X_combined)
    
    log.info(f"Final feature count: {X_final.shape[1]}")
    
    n_keep = int(np.sum(y_ava == 1))
    n_trash = int(np.sum(y_ava == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    log.info(f"\nStarting Optuna optimization ({args.n_trials} trials)...")
    log.info("This may take a while...")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_final, y_ava, scale_pos_weight),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    log.info("\n" + "="*80)
    log.info("OPTIMIZATION RESULTS")
    log.info("="*80)
    log.info(f"Best CV accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
    log.info(f"Best parameters:")
    for key, value in study.best_params.items():
        log.info(f"  {key}: {value}")
    
    # Save results
    results_path = cache_dir / "catboost_ava_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_cv_accuracy": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": args.n_trials,
        }, f, indent=2)
    
    log.info(f"\nResults saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

