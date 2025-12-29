#!/usr/bin/env python3
"""Train the optimal model: CatBoost with best 19 features and tuned hyperparameters.

Usage:
    poetry run python scripts/train_optimal_model.py [IMAGE_DIR] [--model-path PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("train_optimal")


def get_best_feature_indices() -> list[int]:
    """Get best 19-feature set from evaluation results."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    combo_path = cache_dir / "feature_combination_results.json"
    
    if combo_path.exists():
        with open(combo_path) as f:
            data = json.load(f)
            best_config = data.get("best_config")
            if best_config and "feature_indices" in best_config:
                indices = best_config["feature_indices"]
                log.info(f"Loaded best feature set: {len(indices)} features (accuracy={best_config.get('metrics', {}).get('accuracy', 0):.2%})")
                return indices
    
    log.warning("Best feature set not found, using all features")
    from src.features import FEATURE_COUNT
    return list(range(FEATURE_COUNT))


def train_optimal_model(image_dir: str, model_path: str):
    """Train CatBoost with best features and hyperparameters."""
    log.info("="*80)
    log.info("TRAINING OPTIMAL MODEL")
    log.info("="*80)
    log.info(f"Image directory: {image_dir}")
    log.info(f"Model path: {model_path}")
    
    # Load dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info(f"\nLoading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    log.info(f"  Keep: {np.sum(y == 1)}, Trash: {np.sum(y == 0)}")
    
    if len(y) < 20:
        log.error("Insufficient labeled data")
        return 1
    
    # Get best feature set
    feature_indices = get_best_feature_indices()
    X_selected = X[:, feature_indices]
    log.info(f"\nUsing {len(feature_indices)} selected features")
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    
    # Load best hyperparameters
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    # Compute class weights
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Map XGBoost params to CatBoost params
    cb_params = {
        "iterations": xgb_params.get("n_estimators", 300),
        "learning_rate": xgb_params.get("learning_rate", 0.2),
        "depth": xgb_params.get("max_depth", 5),
        "l2_leaf_reg": xgb_params.get("reg_lambda", 5.0),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    
    log.info(f"\nCatBoost parameters: {cb_params}")
    
    # Train CatBoost
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available. Install with: poetry add catboost")
        return 1
    
    log.info("\nTraining CatBoost model...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(**cb_params)),
    ])
    
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')
    
    log.info("\n" + "="*80)
    log.info("TEST SET RESULTS")
    log.info("="*80)
    log.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1:        {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Save model with metadata
    import joblib
    from src.model_version import create_model_metadata
    from src.features import FEATURE_COUNT, USE_FULL_FEATURES
    
    metadata = create_model_metadata(
        feature_count=FEATURE_COUNT,
        feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
        params=cb_params,
        n_samples=len(y_train),
    )
    metadata["feature_indices"] = feature_indices
    metadata["model_type"] = "CatBoost"
    metadata["test_accuracy"] = float(accuracy)
    metadata["test_precision"] = float(precision)
    metadata["test_recall"] = float(recall)
    metadata["test_f1"] = float(f1)
    metadata["test_roc_auc"] = float(roc_auc) if not np.isnan(roc_auc) else None
    
    model_data = {
        "__metadata__": metadata,
        "model": clf,
        "feature_indices": feature_indices,
        "feature_length": len(feature_indices),
        "n_samples": len(y_train),
        "n_keep": n_keep,
        "n_trash": n_trash,
        "filenames": filenames,
        "precision": float(precision),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "f1": float(f1),
    }
    
    joblib.dump(model_data, model_path)
    log.info(f"\nModel saved to: {model_path}")
    
    # Save results summary
    results_path = Path(__file__).resolve().parent.parent / ".cache" / "optimal_model_results.json"
    results = {
        "model_type": "CatBoost",
        "n_features": len(feature_indices),
        "feature_indices": feature_indices,
        "hyperparameters": cb_params,
        "test_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        },
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to: {results_path}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train optimal model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--model-path", default=None, help="Model output path")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    model_path = args.model_path or os.path.expanduser("~/.photo-derush-optimal-model.joblib")
    
    return train_optimal_model(image_dir, model_path)


if __name__ == "__main__":
    raise SystemExit(main())

