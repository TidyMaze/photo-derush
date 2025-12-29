#!/usr/bin/env python3
"""Train the best ensemble model and save it as the default model.

Usage:
    poetry run python scripts/train_best_ensemble_model.py [IMAGE_DIR] [--model-path PATH]
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.training_core import DEFAULT_MODEL_PATH, StackingEnsemble
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("train_best_ensemble")


def get_best_feature_indices():
    """Get best feature indices from feature combination results."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    combo_path = cache_dir / "feature_combination_results.json"
    
    if combo_path.exists():
        try:
            with open(combo_path) as f:
                combo_data = json.load(f)
                best_config = combo_data.get("best_config")
                if best_config:
                    return best_config.get("feature_indices", list(range(FEATURE_COUNT)))
        except Exception:
            pass
    
    # Default: top 20 features (from previous study)
    return list(range(min(20, FEATURE_COUNT)))


def train_ensemble_model(image_dir: str, model_path: str, repo: RatingsTagsRepository):
    """Train the best ensemble model and save it."""
    log.info("="*60)
    log.info("TRAINING BEST ENSEMBLE MODEL")
    log.info("="*60)
    
    # Build dataset
    log.info("Building dataset...")
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)
    
    if len(y) < 20:
        log.error(f"Insufficient labeled data: {len(y)} samples")
        return False
    
    log.info(f"Dataset: {len(y)} samples ({np.sum(y==1)} keep, {np.sum(y==0)} trash)")
    
    # Get best feature indices
    feature_indices = get_best_feature_indices()
    log.info(f"Using {len(feature_indices)} features (best feature set)")
    
    X_subset = X[:, feature_indices]
    
    # Fixed split (same as benchmark)
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X_subset, y, filenames, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Load best hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    log.info(f"Using tuned hyperparameters: {list(xgb_params.keys())}")
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Create base estimators
    log.info("Creating ensemble: XGBoost + Random Forest...")
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        **xgb_params,
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    # Stacking ensemble (best method: 77.78% accuracy)
    log.info("Using stacking ensemble (best method)...")
    
    # Create base models with scalers
    xgb_base = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb_clf),
    ])
    
    rf_base = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", rf_clf),
    ])
    
    # Train base models
    log.info("Training base models...")
    xgb_base.fit(X_train, y_train)
    rf_base.fit(X_train, y_train)
    
    # Get meta-features (predictions from base models)
    log.info("Generating meta-features...")
    xgb_proba = xgb_base.predict_proba(X_train)[:, 1]
    rf_proba = rf_base.predict_proba(X_train)[:, 1]
    meta_X = np.column_stack([xgb_proba, rf_proba])
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(meta_X, y_train)
    
    # Create stacking ensemble
    ensemble = StackingEnsemble([xgb_base, rf_base], meta_learner)
    ensemble.fit(X_train, y_train)  # Train meta-learner (already done above, but ensures consistency)
    
    clf = ensemble
    
    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    log.info(f"Train accuracy: {train_acc:.4f}")
    log.info(f"Test accuracy: {test_acc:.4f}")
    
    # Get feature importances from XGBoost
    feature_importances = None
    try:
        xgb_model = xgb_base.named_steps["xgb"]
        importances = xgb_model.feature_importances_
        feature_importances = [(i, float(imp)) for i, imp in enumerate(importances)]
        feature_importances.sort(key=lambda x: x[1], reverse=True)
    except Exception:
        pass
    
    # Save model
    log.info(f"Saving model to {model_path}...")
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Prepare model data (compatible with inference.py format)
    model_data = {
        "model": ensemble,  # Stacking ensemble
        "scaler": StandardScaler(),  # Base models have their own scalers
        "feature_length": len(feature_indices),
        "feature_indices": feature_indices,  # Store which features are used
        "feature_importances": feature_importances,
        "__metadata__": {
            "model_type": "stacking_xgb_rf",
            "feature_count": len(feature_indices),
            "feature_mode": "FAST",
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "n_samples": len(y),
            "n_keep": int(n_keep),
            "n_trash": int(n_trash),
            "hyperparameters": xgb_params,
        },
    }
    
    joblib.dump(model_data, model_path)
    log.info(f"✅ Model saved successfully!")
    
    log.info("\n" + "="*60)
    log.info("MODEL TRAINING COMPLETE")
    log.info("="*60)
    log.info(f"Model path: {model_path}")
    log.info(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    log.info(f"Features: {len(feature_indices)}")
    log.info(f"Model type: Stacking Ensemble (XGBoost + Random Forest)")
    log.info("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train best ensemble model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to save model")
    args = parser.parse_args()
    
    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    model_path = os.path.expanduser(args.model_path)
    
    # Initialize repository
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)
    
    # Train model
    success = train_ensemble_model(image_dir, model_path, repo)
    
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

