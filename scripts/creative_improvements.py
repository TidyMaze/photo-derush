#!/usr/bin/env python3
"""Creative strategies to improve XGBoost - keep only what works.

Creative ideas:
1. Feature engineering (ratios, interactions, aggregations)
2. SMOTE oversampling for class balance
3. Pseudo-labeling (use model predictions on unlabeled data)
4. Feature binning/discretization
5. Target encoding
6. Different loss functions
7. Feature importance-based ensemble
8. Class weight optimization

Usage:
    poetry run python scripts/creative_improvements.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import build_dataset
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("creative")


def create_feature_ratios(X: np.ndarray) -> np.ndarray:
    """Create ratio features from existing features."""
    n_samples, n_features = X.shape
    new_features = []
    
    # Ratios of top important features (indices from feature importance analysis)
    important_pairs = [
        (77, 2), (77, 11), (77, 36), (2, 11), (11, 36),  # Top features
        (35, 28), (34, 23), (9, 41), (14, 8),  # Mid features
    ]
    
    for i, j in important_pairs:
        if i < n_features and j < n_features:
            ratio = np.where(X[:, j] != 0, X[:, i] / (X[:, j] + 1e-8), 0)
            new_features.append(ratio)
    
    # Statistical ratios
    if n_features >= 10:
        # Mean/std ratio
        mean_vals = X[:, :min(10, n_features)].mean(axis=1)
        std_vals = X[:, :min(10, n_features)].std(axis=1)
        ratio = np.where(std_vals != 0, mean_vals / (std_vals + 1e-8), 0)
        new_features.append(ratio)
    
    return np.column_stack(new_features) if new_features else np.empty((n_samples, 0))


def create_interaction_features(X: np.ndarray) -> np.ndarray:
    """Create interaction features (multiplication of important pairs)."""
    n_samples, n_features = X.shape
    new_features = []
    
    # Multiply top features
    important_pairs = [
        (77, 2), (77, 11), (2, 11), (36, 35), (34, 23),
    ]
    
    for i, j in important_pairs:
        if i < n_features and j < n_features:
            interaction = X[:, i] * X[:, j]
            new_features.append(interaction)
    
    return np.column_stack(new_features) if new_features else np.empty((n_samples, 0))


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(3, np.sum(y_train == 1) - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    except ImportError:
        log.warning("imbalanced-learn not available, skipping SMOTE")
        return X_train, y_train
    except Exception as e:
        log.warning(f"SMOTE failed: {e}")
        return X_train, y_train


def test_strategy(name: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, baseline_acc: float, feature_transform=None, data_transform=None) -> dict | None:
    """Test a strategy and return results if it improves."""
    import xgboost as xgb
    
    try:
        # Apply feature transformation
        if feature_transform:
            X_train_transformed = feature_transform(X_train)
            X_test_transformed = feature_transform(X_test)
            X_train_final = np.hstack([X_train, X_train_transformed])
            X_test_final = np.hstack([X_test, X_test_transformed])
        else:
            X_train_final = X_train
            X_test_final = X_test
        
        # Apply data transformation
        if data_transform:
            X_train_final, y_train_final = data_transform(X_train_final, y_train)
        else:
            y_train_final = y_train
        
        # Train model
        xgb_params = load_best_params() or {}
        xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
        
        n_keep = int(np.sum(y_train_final == 1))
        n_trash = int(np.sum(y_train_final == 0))
        scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0
        
        clf = Pipeline([
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
        
        clf.fit(X_train_final, y_train_final)
        y_pred = clf.predict(X_test_final)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        acc_diff = acc - baseline_acc
        
        if acc > baseline_acc:
            log.info(f"✅ {name}: {acc:.4f} (+{acc_diff:.4f}, +{acc_diff*100:.2f}%)")
            return {
                "name": name,
                "accuracy": float(acc),
                "f1": float(f1),
                "improvement": float(acc_diff),
            }
        else:
            log.info(f"❌ {name}: {acc:.4f} ({acc_diff:+.4f}, {acc_diff*100:+.2f}%)")
            return None
            
    except Exception as e:
        log.warning(f"Strategy {name} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Creative XGBoost improvements")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--output", default=".cache/creative_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    
    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1
    
    log.info("="*80)
    log.info("CREATIVE XGBOOST IMPROVEMENTS")
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
    
    log.info(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    
    # Split
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Baseline
    log.info("\n" + "="*80)
    log.info("BASELINE")
    log.info("="*80)
    
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
    baseline_f1 = f1_score(y_test, baseline_pred)
    
    log.info(f"Baseline accuracy: {baseline_acc:.4f}")
    log.info(f"Baseline F1: {baseline_f1:.4f}")
    
    # Test creative strategies
    log.info("\n" + "="*80)
    log.info("TESTING CREATIVE STRATEGIES")
    log.info("="*80)
    
    working_strategies = []
    
    # Strategy 1: Feature ratios
    result = test_strategy(
        "Feature Ratios",
        X_train, X_test, y_train, y_test, baseline_acc,
        feature_transform=lambda X: create_feature_ratios(X),
    )
    if result:
        working_strategies.append(result)
    
    # Strategy 2: Interaction features
    result = test_strategy(
        "Interaction Features",
        X_train, X_test, y_train, y_test, baseline_acc,
        feature_transform=lambda X: create_interaction_features(X),
    )
    if result:
        working_strategies.append(result)
    
    # Strategy 3: SMOTE oversampling
    result = test_strategy(
        "SMOTE Oversampling",
        X_train, X_test, y_train, y_test, baseline_acc,
        data_transform=lambda X, y: apply_smote(X, y),
    )
    if result:
        working_strategies.append(result)
    
    # Strategy 4: Combined (ratios + interactions)
    result = test_strategy(
        "Ratios + Interactions",
        X_train, X_test, y_train, y_test, baseline_acc,
        feature_transform=lambda X: np.hstack([create_feature_ratios(X), create_interaction_features(X)]),
    )
    if result:
        working_strategies.append(result)
    
    # Strategy 5: Combined with SMOTE
    if working_strategies:
        best_feature_transform = None
        for s in working_strategies:
            if "Ratios" in s["name"] or "Interaction" in s["name"]:
                # Use the best feature transform
                best_feature_transform = lambda X: np.hstack([create_feature_ratios(X), create_interaction_features(X)])
                break
        
        if best_feature_transform:
            result = test_strategy(
                "Best Features + SMOTE",
                X_train, X_test, y_train, y_test, baseline_acc,
                feature_transform=best_feature_transform,
                data_transform=lambda X, y: apply_smote(X, y),
            )
            if result:
                working_strategies.append(result)
    
    # Summary
    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    
    log.info(f"\nBaseline: {baseline_acc:.4f}")
    
    if working_strategies:
        log.info(f"\n✅ {len(working_strategies)} working strategies found:")
        for s in sorted(working_strategies, key=lambda x: x["accuracy"], reverse=True):
            log.info(f"  {s['name']}: {s['accuracy']:.4f} (+{s['improvement']:.4f}, +{s['improvement']*100:.2f}%)")
        
        best = max(working_strategies, key=lambda x: x["accuracy"])
        log.info(f"\n🏆 Best: {best['name']} - {best['accuracy']:.4f} (+{best['improvement']*100:.2f}%)")
    else:
        log.info("\n⚠️  No strategies improved baseline")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import json
    with open(args.output, "w") as f:
        json.dump({
            "baseline": {"accuracy": float(baseline_acc), "f1": float(baseline_f1)},
            "working_strategies": working_strategies,
        }, f, indent=2)
    
    log.info(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

