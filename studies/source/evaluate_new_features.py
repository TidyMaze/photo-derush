#!/usr/bin/env python3
"""Evaluate impact of new photography features on model accuracy."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT
from src.model import RatingsTagsRepository
from src.training_core import _fit_model, _get_feature_importances, _compute_metrics
from src.tuning import load_best_params
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("eval_features")

# New feature indices (added at end: 71-76)
NEW_FEATURE_INDICES = {
    71: "Subject Isolation",
    72: "Golden Hour",
    73: "Lighting Quality",
    74: "Color Harmony",
    75: "Sky/Ground Ratio",
    76: "Motion Blur",
}


def main():
    image_dir = os.path.expanduser("~/Pictures/photo-dataset")
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)

    log.info("Loading dataset...")
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 10:
        log.error(f"Insufficient labeled data: {len(y)} samples")
        return 1

    log.info(f"Dataset: {len(y)} samples, {FEATURE_COUNT} features")

    # Fixed train/test split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42, stratify=y
    )

    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train model with ALL features (including new ones)
    log.info("\n" + "="*80)
    log.info("Training model with ALL features (including new photography features)...")
    log.info("="*80)

    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
    
    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    clf_all = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=xgb_params.get("n_estimators", 100),
            **{k: v for k, v in xgb_params.items() if k != "n_estimators"},
        )),
    ])
    
    _fit_model(clf_all, X_train, y_train, len(y_train), early_stopping_rounds=None)
    feature_importances_all = _get_feature_importances(clf_all)
    
    # Compute metrics for stats
    precision, roc_auc, f1, confusion = _compute_metrics(clf_all, X_train, y_train)
    stats_all = {
        "feature_importances": feature_importances_all,
        "precision": precision,
        "roc_auc": roc_auc,
        "f1": f1,
    }
    
    # Benchmark with all features
    result_all = benchmark_model(
        clf_all,
        X_train,
        X_test,
        y_train,
        y_test,
        filenames_test,
        model_name="XGBoost (All Features)",
        feature_importances=feature_importances_all,
        feature_count=FEATURE_COUNT,
    )

    accuracy_all = result_all.metrics["accuracy"]
    log.info(f"\n✅ Accuracy with ALL features: {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")

    # Train model WITHOUT new features (only first 71 features)
    log.info("\n" + "="*80)
    log.info("Training model WITHOUT new features (baseline: 71 features)...")
    log.info("="*80)

    X_train_baseline = X_train[:, :71]
    X_test_baseline = X_test[:, :71]

    clf_baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=xgb_params.get("n_estimators", 100),
            **{k: v for k, v in xgb_params.items() if k != "n_estimators"},
        )),
    ])
    
    _fit_model(clf_baseline, X_train_baseline, y_train, len(y_train), early_stopping_rounds=None)
    feature_importances_baseline = _get_feature_importances(clf_baseline)
    
    precision_b, roc_auc_b, f1_b, confusion_b = _compute_metrics(clf_baseline, X_train_baseline, y_train)
    stats_baseline = {
        "feature_importances": feature_importances_baseline,
        "precision": precision_b,
        "roc_auc": roc_auc_b,
        "f1": f1_b,
    }

    result_baseline = benchmark_model(
        clf_baseline,
        X_train_baseline,
        X_test_baseline,
        y_train,
        y_test,
        filenames_test,
        model_name="XGBoost (Baseline 71 Features)",
        feature_importances=feature_importances_baseline,
        feature_count=71,
    )

    accuracy_baseline = result_baseline.metrics["accuracy"]
    log.info(f"\n✅ Accuracy BASELINE (71 features): {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")

    # Calculate improvement
    improvement = accuracy_all - accuracy_baseline
    improvement_pct = (improvement / accuracy_baseline * 100) if accuracy_baseline > 0 else 0.0

    log.info("\n" + "="*80)
    log.info("RESULTS SUMMARY")
    log.info("="*80)
    log.info(f"Baseline accuracy (71 features): {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
    log.info(f"Accuracy with new features (77 features): {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")
    log.info(f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

    # Analyze which new features are most important
    log.info("\n" + "="*80)
    log.info("NEW FEATURE IMPORTANCE ANALYSIS")
    log.info("="*80)

    # Get ALL feature importances, not just top 10
    try:
        xgb_model = clf_all.named_steps.get("xgb")
        if xgb_model and hasattr(xgb_model, "feature_importances_"):
            all_importances = xgb_model.feature_importances_
            # Extract new feature importances
            new_feature_importance = {}
            for feat_idx in NEW_FEATURE_INDICES:
                if feat_idx < len(all_importances):
                    new_feature_importance[NEW_FEATURE_INDICES[feat_idx]] = float(all_importances[feat_idx])
        else:
            new_feature_importance = {}
            for feat_idx, importance in feature_importances_all:
                if feat_idx in NEW_FEATURE_INDICES:
                    new_feature_importance[NEW_FEATURE_INDICES[feat_idx]] = importance
    except Exception as e:
        log.warning(f"Could not extract all importances: {e}")
        new_feature_importance = {}
        for feat_idx, importance in feature_importances_all:
            if feat_idx in NEW_FEATURE_INDICES:
                new_feature_importance[NEW_FEATURE_INDICES[feat_idx]] = importance

    if new_feature_importance:
        sorted_new = sorted(new_feature_importance.items(), key=lambda x: x[1], reverse=True)
        log.info("\nNew features ranked by importance:")
        for rank, (name, imp) in enumerate(sorted_new, 1):
            log.info(f"  {rank}. {name}: {imp*100:.3f}%")
        
        # Check if any new features are in top 20
        try:
            xgb_model = clf_all.named_steps.get("xgb")
            if xgb_model and hasattr(xgb_model, "feature_importances_"):
                all_importances = xgb_model.feature_importances_
                top_20_indices = np.argsort(all_importances)[::-1][:20]
                new_in_top20 = [NEW_FEATURE_INDICES[idx] for idx in top_20_indices if idx in NEW_FEATURE_INDICES]
            else:
                top_20_indices = [idx for idx, _ in feature_importances_all[:20]]
                new_in_top20 = [NEW_FEATURE_INDICES[idx] for idx in top_20_indices if idx in NEW_FEATURE_INDICES]
        except Exception:
            top_20_indices = [idx for idx, _ in feature_importances_all[:20]]
            new_in_top20 = [NEW_FEATURE_INDICES[idx] for idx in top_20_indices if idx in NEW_FEATURE_INDICES]
        
        if new_in_top20:
            log.info(f"\n✨ New features in TOP 20: {', '.join(new_in_top20)}")
        else:
            log.info("\n⚠️  No new features in top 20 (but may still contribute to ensemble)")
    else:
        log.warning("Could not extract new feature importances")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "..", ".cache", "new_features_evaluation.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    results = {
        "baseline_accuracy": float(accuracy_baseline),
        "all_features_accuracy": float(accuracy_all),
        "improvement": float(improvement),
        "improvement_percent": float(improvement_pct),
        "new_feature_importances": {k: float(v) for k, v in new_feature_importance.items()},
        "new_features_in_top20": new_in_top20 if new_feature_importance else [],
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"\n✅ Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

