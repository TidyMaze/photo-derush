#!/usr/bin/env python3
"""More creative strategies - pseudo-labeling, ensemble, class weight optimization."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("creative_v2")


def test_pseudo_labeling(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, X_unlabeled: np.ndarray, baseline_acc: float) -> dict | None:
    """Use pseudo-labeling on unlabeled data."""
    import xgboost as xgb
    
    if len(X_unlabeled) == 0:
        return None
    
    try:
        xgb_params = load_best_params() or {}
        xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
        
        n_keep = int(np.sum(y_train == 1))
        n_trash = int(np.sum(y_train == 0))
        scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
        
        # Train initial model
        clf1 = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                **xgb_params,
            )),
        ])
        clf1.fit(X_train, y_train)
        
        # Get high-confidence predictions on unlabeled data
        y_proba_unlabeled = clf1.predict_proba(X_unlabeled)[:, 1]
        high_conf_mask = (y_proba_unlabeled > 0.9) | (y_proba_unlabeled < 0.1)
        
        if np.sum(high_conf_mask) < 10:
            return None
        
        X_pseudo = X_unlabeled[high_conf_mask]
        y_pseudo = (y_proba_unlabeled[high_conf_mask] > 0.5).astype(int)
        
        # Combine with training data
        X_combined = np.vstack([X_train, X_pseudo])
        y_combined = np.hstack([y_train, y_pseudo])
        
        # Retrain
        n_keep_new = int(np.sum(y_combined == 1))
        n_trash_new = int(np.sum(y_combined == 0))
        scale_pos_weight_new = n_trash_new / n_keep_new if n_keep_new > 0 else 1.0
        
        clf2 = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight_new,
                objective="binary:logistic",
                eval_metric="logloss",
                **xgb_params,
            )),
        ])
        clf2.fit(X_combined, y_combined)
        
        y_pred = clf2.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_diff = acc - baseline_acc
        
        if acc > baseline_acc:
            log.info(f"✅ Pseudo-labeling: {acc:.4f} (+{acc_diff:.4f}, +{acc_diff*100:.2f}%)")
            return {"name": "Pseudo-labeling", "accuracy": float(acc), "improvement": float(acc_diff)}
        else:
            log.info(f"❌ Pseudo-labeling: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
            return None
            
    except Exception as e:
        log.warning(f"Pseudo-labeling failed: {e}")
        return None


def test_class_weight_optimization(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict | None:
    """Optimize class weight."""
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    base_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    best_acc = baseline_acc
    best_weight = base_weight
    
    # Try different weights
    weights = [base_weight * w for w in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
    
    for weight in weights:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=weight,
                objective="binary:logistic",
                eval_metric="logloss",
                **xgb_params,
            )),
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_weight = weight
    
    acc_diff = best_acc - baseline_acc
    
    if best_acc > baseline_acc:
        log.info(f"✅ Class Weight Opt: {best_acc:.4f} (+{acc_diff:.4f}, +{acc_diff*100:.2f}%) weight={best_weight:.3f}")
        return {"name": "Class Weight Optimization", "accuracy": float(best_acc), "improvement": float(acc_diff), "weight": float(best_weight)}
    else:
        log.info(f"❌ Class Weight Opt: {best_acc:.4f} (no improvement)")
        return None


def test_diverse_ensemble(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float) -> dict | None:
    """Ensemble of models with different random seeds."""
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models with different seeds
    models = []
    for seed in [42, 123, 456, 789, 999]:
        clf = xgb.XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            **xgb_params,
        )
        clf.fit(X_train_scaled, y_train)
        models.append(("xgb_" + str(seed), clf))
    
    # Voting ensemble
    ensemble = VotingClassifier(estimators=models, voting="soft")
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    acc_diff = acc - baseline_acc
    
    if acc > baseline_acc:
        log.info(f"✅ Diverse Ensemble: {acc:.4f} (+{acc_diff:.4f}, +{acc_diff*100:.2f}%)")
        return {"name": "Diverse Ensemble", "accuracy": float(acc), "improvement": float(acc_diff)}
    else:
        log.info(f"❌ Diverse Ensemble: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
        return None


def main():
    parser = argparse.ArgumentParser(description="Creative improvements v2")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("CREATIVE IMPROVEMENTS V2")
    log.info("="*80)
    
    # Build dataset
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path) if os.path.exists(repo_path) else None
    
    if repo is None:
        log.error("No repository found")
        return 1
    
    log.info(f"\nLoading dataset from {image_dir}...")
    X, y, filenames = build_dataset(image_dir, repo=repo)
    X = np.array(X)
    y = np.array(y)
    
    # Get unlabeled data
    all_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    unlabeled_files = []
    for fname in all_files:
        state = repo.get_state(fname)
        if state not in ("keep", "trash"):
            unlabeled_files.append(fname)
    
    log.info(f"Dataset: {len(y)} labeled, {len(unlabeled_files)} unlabeled")
    
    # Split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Get unlabeled features (if any)
    X_unlabeled = np.empty((0, X.shape[1]))
    if unlabeled_files and len(unlabeled_files) > 10:
        try:
            from src.features import batch_extract_features
            unlabeled_paths = [os.path.join(image_dir, f) for f in unlabeled_files[:100]]  # Limit to 100
            unlabeled_features = batch_extract_features(unlabeled_paths)
            X_unlabeled = np.array([f for f in unlabeled_features if f is not None])
            log.info(f"Extracted {len(X_unlabeled)} unlabeled samples")
        except Exception as e:
            log.warning(f"Could not extract unlabeled features: {e}")
    
    # Baseline
    import xgboost as xgb
    
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
    
    baseline_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            **xgb_params,
        )),
    ])
    
    baseline_clf.fit(X_train, y_train)
    baseline_pred = baseline_clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    log.info(f"\nBaseline accuracy: {baseline_acc:.4f}")
    
    # Test strategies
    log.info("\n" + "="*80)
    log.info("TESTING STRATEGIES")
    log.info("="*80)
    
    working = []
    
    # Strategy 1: Class weight optimization
    result = test_class_weight_optimization(X_train, X_test, y_train, y_test, baseline_acc)
    if result:
        working.append(result)
    
    # Strategy 2: Diverse ensemble
    result = test_diverse_ensemble(X_train, X_test, y_train, y_test, baseline_acc)
    if result:
        working.append(result)
    
    # Strategy 3: Pseudo-labeling
    if len(X_unlabeled) > 0:
        result = test_pseudo_labeling(X_train, X_test, y_train, y_test, X_unlabeled, baseline_acc)
        if result:
            working.append(result)
    
    # Summary
    log.info("\n" + "="*80)
    log.info("RESULTS")
    log.info("="*80)
    
    if working:
        log.info(f"\n✅ {len(working)} working strategies:")
        for s in sorted(working, key=lambda x: x["accuracy"], reverse=True):
            log.info(f"  {s['name']}: {s['accuracy']:.4f} (+{s['improvement']*100:.2f}%)")
        best = max(working, key=lambda x: x["accuracy"])
        log.info(f"\n🏆 Best: {best['name']} - {best['accuracy']:.4f} (+{best['improvement']*100:.2f}%)")
    else:
        log.info("\n⚠️  No strategies improved baseline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

