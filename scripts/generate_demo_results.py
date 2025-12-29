#!/usr/bin/env python3
"""Generate demo benchmark results for visualization purposes.

Usage:
    poetry run python scripts/generate_demo_results.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("demo_results")

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def generate_demo_results():
    """Generate realistic demo benchmark results."""
    import random
    from datetime import datetime

    random.seed(42)

    # Baseline
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "XGBoost",
        "model_path": None,
        "hyperparameters": {"n_estimators": 200, "learning_rate": 0.1},
        "dataset_info": {"train_size": 800, "test_size": 200, "train_keep": 400, "train_trash": 400},
        "metrics": {
            "accuracy": 0.845,
            "precision": 0.832,
            "recall": 0.860,
            "f1": 0.846,
            "roc_auc": 0.912,
            "brier_score": 0.112,
            "confusion_tn": 85,
            "confusion_fp": 15,
            "confusion_fn": 14,
            "confusion_tp": 86,
        },
        "feature_importances": [(12, 0.15), (5, 0.12), (8, 0.10), (3, 0.09), (20, 0.08)],
        "test_filenames": [],
        "train_size": 800,
        "test_size": 200,
        "feature_count": 71,
        "feature_mode": "FAST",
    }

    # Algorithm comparison
    algorithms = [
        {"algorithm": "XGBoost", "metrics": {"accuracy": 0.845, "f1": 0.846, "roc_auc": 0.912, "precision": 0.832}},
        {"algorithm": "LightGBM", "metrics": {"accuracy": 0.838, "f1": 0.839, "roc_auc": 0.905, "precision": 0.825}},
        {"algorithm": "CatBoost", "metrics": {"accuracy": 0.841, "f1": 0.842, "roc_auc": 0.908, "precision": 0.828}},
        {"algorithm": "RandomForest", "metrics": {"accuracy": 0.815, "f1": 0.816, "roc_auc": 0.885, "precision": 0.805}},
        {"algorithm": "MLP", "metrics": {"accuracy": 0.802, "f1": 0.803, "roc_auc": 0.872, "precision": 0.795}},
    ]

    # Feature ablation
    feature_ablation = [
        {"group_name": "all", "features_used": 71, "metrics": {"accuracy": 0.845, "f1": 0.846}},
        {"group_name": "no_geometric", "features_used": 67, "metrics": {"accuracy": 0.840, "f1": 0.841}},
        {"group_name": "no_image_stats", "features_used": 63, "metrics": {"accuracy": 0.835, "f1": 0.836}},
        {"group_name": "no_histograms", "features_used": 47, "metrics": {"accuracy": 0.825, "f1": 0.826}},
        {"group_name": "no_quality", "features_used": 65, "metrics": {"accuracy": 0.830, "f1": 0.831}},
        {"group_name": "no_temporal", "features_used": 67, "metrics": {"accuracy": 0.842, "f1": 0.843}},
        {"group_name": "no_exif", "features_used": 46, "metrics": {"accuracy": 0.820, "f1": 0.821}},
    ]

    # Ensemble results
    ensemble_results = [
        {"method": "hard_voting", "metrics": {"accuracy": 0.850, "f1": 0.851, "roc_auc": 0.918}},
        {"method": "soft_voting", "metrics": {"accuracy": 0.852, "f1": 0.853, "roc_auc": 0.920}},
        {"method": "blending_equal", "metrics": {"accuracy": 0.848, "f1": 0.849, "roc_auc": 0.915}},
        {"method": "blending_learned", "metrics": {"accuracy": 0.855, "f1": 0.856, "roc_auc": 0.922}},
        {"method": "stacking", "metrics": {"accuracy": 0.853, "f1": 0.854, "roc_auc": 0.921}},
    ]

    # Calibration results
    calibration_results = [
        {"method": "uncalibrated", "brier_score": 0.112, "ece": 0.045, "metrics": {"accuracy": 0.845}},
        {"method": "platt_scaling", "brier_score": 0.098, "ece": 0.032, "metrics": {"accuracy": 0.845}},
        {"method": "isotonic_regression", "brier_score": 0.095, "ece": 0.028, "metrics": {"accuracy": 0.845}},
        {"method": "temperature_scaling", "brier_score": 0.100, "ece": 0.035, "metrics": {"accuracy": 0.845}},
    ]

    # Class balance results
    class_balance_results = [
        {"method": "baseline", "metrics": {"accuracy": 0.845, "precision": 0.832, "recall": 0.860, "f1": 0.846}},
        {"method": "smote", "metrics": {"accuracy": 0.838, "precision": 0.850, "recall": 0.825, "f1": 0.837}},
        {"method": "adasyn", "metrics": {"accuracy": 0.840, "precision": 0.848, "recall": 0.830, "f1": 0.839}},
        {"method": "undersampling", "metrics": {"accuracy": 0.815, "precision": 0.820, "recall": 0.810, "f1": 0.815}},
        {"method": "cost_sensitive_2.0", "metrics": {"accuracy": 0.842, "precision": 0.835, "recall": 0.855, "f1": 0.845}},
    ]

    # Save files
    baseline_path = CACHE_DIR / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    algo_path = CACHE_DIR / "algorithm_comparison_results.json"
    with open(algo_path, "w") as f:
        json.dump(algorithms, f, indent=2, default=str)

    ablation_path = CACHE_DIR / "feature_ablation_results.json"
    with open(ablation_path, "w") as f:
        json.dump(feature_ablation, f, indent=2, default=str)

    ensemble_path = CACHE_DIR / "ensemble_results.json"
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_results, f, indent=2, default=str)

    calib_path = CACHE_DIR / "calibration_results.json"
    with open(calib_path, "w") as f:
        json.dump(calibration_results, f, indent=2, default=str)

    balance_path = CACHE_DIR / "class_balance_results.json"
    with open(balance_path, "w") as f:
        json.dump(class_balance_results, f, indent=2, default=str)

    log.info("Generated demo benchmark results")
    return True


if __name__ == "__main__":
    generate_demo_results()

