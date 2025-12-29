#!/usr/bin/env python3
"""Train the final optimal model: CatBoost with best features, testing both all features and 19-feature set.

Usage:
    poetry run python scripts/train_final_optimal_model.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("train_final")


def get_best_feature_indices() -> list[int]:
    """Get best 19-feature set from evaluation results."""
    cache_dir = Path(__file__).resolve().parent.parent / ".cache"
    combo_path = cache_dir / "feature_combination_results.json"
    
    if combo_path.exists():
        with open(combo_path) as f:
            data = json.load(f)
            best_config = data.get("best_config")
            if best_config and "feature_indices" in best_config:
                return best_config["feature_indices"]
    
    return None


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_indices, config_name):
    """Train CatBoost and evaluate."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        log.error("CatBoost not available")
        return None
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    # Use settings that achieved 84% in alternative models test
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostClassifier(
            random_state=42,
            thread_count=-1,
            scale_pos_weight=scale_pos_weight,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            iterations=200,  # As used in test_alternative_models
            learning_rate=0.1,  # As used in test_alternative_models
            depth=6,  # As used in test_alternative_models
        )),
    ])
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        "config": config_name,
        "n_features": len(feature_indices) if feature_indices else X_train.shape[1],
        "feature_indices": feature_indices,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "model": clf,
    }


def main():
    parser = argparse.ArgumentParser(description="Train final optimal model")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    args = parser.parse_args()
    
    image_dir = args.image_dir or os.path.expanduser("~/Pictures/photo-dataset")
    image_dir = os.path.expanduser(image_dir)
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("TRAINING FINAL OPTIMAL MODEL")
    log.info("="*80)
    
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
    
    # Fixed stratified split (same as evaluation scripts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    log.info(f"Train: {len(y_train)} samples, Test: {len(y_test)} samples")
    
    results = []
    
    # Test 1: All features (as in alternative models test)
    log.info("\n" + "="*80)
    log.info("CONFIG 1: CatBoost with ALL features")
    log.info("="*80)
    result1 = train_and_evaluate(X_train, X_test, y_train, y_test, None, "all_features")
    if result1:
        results.append(result1)
        m = result1["metrics"]
        log.info(f"Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        log.info(f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}, ROC-AUC: {m['roc_auc']:.4f}")
    
    # Test 2: Best 19 features
    feature_indices = get_best_feature_indices()
    if feature_indices:
        log.info("\n" + "="*80)
        log.info(f"CONFIG 2: CatBoost with {len(feature_indices)} best features")
        log.info("="*80)
        X_train_selected = X_train[:, feature_indices]
        X_test_selected = X_test[:, feature_indices]
        result2 = train_and_evaluate(X_train_selected, X_test_selected, y_train, y_test, feature_indices, "best_19_features")
        if result2:
            results.append(result2)
            m = result2["metrics"]
            log.info(f"Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
            log.info(f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}, ROC-AUC: {m['roc_auc']:.4f}")
    
    # Pick best
    if results:
        best = max(results, key=lambda r: r["metrics"]["accuracy"])
        log.info("\n" + "="*80)
        log.info("BEST CONFIGURATION")
        log.info("="*80)
        log.info(f"Config: {best['config']}")
        log.info(f"Features: {best['n_features']}")
        m = best["metrics"]
        log.info(f"Accuracy: {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
        log.info(f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}, ROC-AUC: {m['roc_auc']:.4f}")
        
        # Save best model
        import joblib
        from src.model_version import create_model_metadata
        from src.features import FEATURE_COUNT, USE_FULL_FEATURES
        
        model_path = os.path.expanduser("~/.photo-derush-final-model.joblib")
        
        metadata = create_model_metadata(
            feature_count=FEATURE_COUNT,
            feature_mode="FULL" if USE_FULL_FEATURES else "FAST",
            params={"iterations": 200, "learning_rate": 0.1, "depth": 6},
            n_samples=len(y_train),
        )
        if best.get("feature_indices"):
            metadata["feature_indices"] = best["feature_indices"]
        metadata["model_type"] = "CatBoost"
        metadata["test_accuracy"] = m["accuracy"]
        metadata["test_precision"] = m["precision"]
        metadata["test_recall"] = m["recall"]
        metadata["test_f1"] = m["f1"]
        metadata["test_roc_auc"] = m["roc_auc"]
        
        model_data = {
            "__metadata__": metadata,
            "model": best["model"],
            "feature_indices": best.get("feature_indices"),
            "feature_length": best["n_features"],
            "n_samples": len(y_train),
            "n_keep": int(np.sum(y_train == 1)),
            "n_trash": int(np.sum(y_train == 0)),
            "precision": m["precision"],
            "roc_auc": m["roc_auc"],
            "f1": m["f1"],
        }
        
        joblib.dump(model_data, model_path)
        log.info(f"\nBest model saved to: {model_path}")
        
        # Save results
        results_path = Path(__file__).resolve().parent.parent / ".cache" / "final_optimal_model_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "best_config": best["config"],
                "best_n_features": best["n_features"],
                "best_metrics": best["metrics"],
                "all_results": [
                    {
                        "config": r["config"],
                        "n_features": r["n_features"],
                        "metrics": r["metrics"],
                    }
                    for r in results
                ],
            }, f, indent=2)
        
        log.info(f"Results saved to: {results_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

