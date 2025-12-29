#!/usr/bin/env python3
"""Experiment with ensemble methods: voting, stacking, blending.

Usage:
    poetry run python scripts/ensemble_experiments.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("ensemble")

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class BlendingEnsemble:
    """Weighted average ensemble with learned weights."""

    def __init__(self, base_models: list[Pipeline], weights: list[float] | None = None):
        self.base_models = base_models
        self.weights = weights if weights else [1.0 / len(base_models)] * len(base_models)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all base models."""
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predictions."""
        probas = []
        for model, weight in zip(self.base_models, self.weights):
            proba = model.predict_proba(X)[:, 1]
            probas.append(proba * weight)
        return np.column_stack([1 - np.sum(probas, axis=0), np.sum(probas, axis=0)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted probabilities."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""

    def __init__(self, base_models: list[Pipeline], meta_learner=None):
        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(random_state=42, max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train base models and meta-learner."""
        # Train base models
        for model in self.base_models:
            model.fit(X, y)

        # Get base model predictions for meta-features
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba)
        meta_X = np.column_stack(meta_features)

        # Train meta-learner
        self.meta_learner.fit(meta_X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using meta-learner."""
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba)
        meta_X = np.column_stack(meta_features)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using meta-learner."""
        return self.meta_learner.predict(self._get_meta_features(X))

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base models."""
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba)
        return np.column_stack(meta_features)


def create_base_models(X_train: np.ndarray, y_train: np.ndarray) -> list[tuple[str, Pipeline]]:
    """Create base models for ensemble."""
    models = []

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    if xgb:
        xgb_params = load_best_params() or {}
        xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}
        models.append(
            (
                "xgb",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "xgb",
                            xgb.XGBClassifier(
                                random_state=42,
                                n_jobs=-1,
                                scale_pos_weight=scale_pos_weight,
                                objective="binary:logistic",
                                eval_metric="logloss",
                                **xgb_params,
                            ),
                        ),
                    ]
                ),
            )
        )

    if lgb:
        models.append(
            (
                "lgb",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "lgb",
                            lgb.LGBMClassifier(
                                random_state=42,
                                n_jobs=-1,
                                scale_pos_weight=scale_pos_weight,
                                objective="binary",
                                metric="binary_logloss",
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
            )
        )

    models.append(
        (
            "rf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=10,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced" if n_keep > 0 and n_trash > 0 else None,
                        ),
                    ),
                ]
            ),
        )
    )

    return models


def learn_blending_weights(
    base_models: list[Pipeline], X_val: np.ndarray, y_val: np.ndarray
) -> list[float]:
    """Learn optimal blending weights on validation set."""
    # Get predictions from each model
    predictions = []
    for model in base_models:
        proba = model.predict_proba(X_val)[:, 1]
        predictions.append(proba)

    predictions = np.array(predictions).T  # Shape: (n_samples, n_models)

    # Optimize weights to minimize log loss
    from scipy.optimize import minimize

    def objective(weights):
        weighted_proba = np.dot(predictions, weights)
        weighted_proba = np.clip(weighted_proba, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y_val * np.log(weighted_proba) + (1 - y_val) * np.log(1 - weighted_proba))
        return log_loss

    # Constraints: weights sum to 1, all >= 0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1) for _ in range(len(base_models))]
    initial_weights = np.ones(len(base_models)) / len(base_models)

    result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x.tolist() if result.success else [1.0 / len(base_models)] * len(base_models)


def main():
    parser = argparse.ArgumentParser(description="Ensemble experiments")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")

    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1

    log.info(f"Loading dataset from {image_dir}")

    # Initialize repository
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)

    # Build dataset
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 30:
        log.error(f"Insufficient labeled data: {len(y)} samples (need at least 30)")
        return 1

    # Split: train (60%), val (20%), test (20%)
    X_train, X_rest, y_train, y_rest, filenames_train, filenames_rest = train_test_split(
        X, y, filenames, test_size=0.4, stratify=y, random_state=args.random_state
    )
    X_val, X_test, y_val, y_test, filenames_val, filenames_test = train_test_split(
        X_rest, y_rest, filenames_rest, test_size=0.5, stratify=y_rest, random_state=args.random_state
    )

    log.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Create base models
    base_model_configs = create_base_models(X_train, y_train)
    base_models = [model for _, model in base_model_configs]
    base_names = [name for name, _ in base_model_configs]

    log.info(f"Base models: {base_names}")

    # Train base models
    log.info("Training base models...")
    for model in base_models:
        model.fit(X_train, y_train)

    results = []

    # 1. Hard Voting
    log.info("\n=== Hard Voting Ensemble ===")
    try:
        voting_hard = VotingClassifier(
            estimators=[(name, model) for name, model in zip(base_names, base_models)], voting="hard"
        )
        voting_hard.fit(X_train, y_train)
        benchmark_result = benchmark_model(
            voting_hard,
            X_train,
            X_test,
            y_train,
            y_test,
            filenames_test,
            "HardVoting",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "hard_voting", "metrics": benchmark_result.metrics})
        log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.error(f"Hard voting failed: {e}")

    # 2. Soft Voting
    log.info("\n=== Soft Voting Ensemble ===")
    try:
        voting_soft = VotingClassifier(
            estimators=[(name, model) for name, model in zip(base_names, base_models)], voting="soft"
        )
        voting_soft.fit(X_train, y_train)
        benchmark_result = benchmark_model(
            voting_soft,
            X_train,
            X_test,
            y_train,
            y_test,
            filenames_test,
            "SoftVoting",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "soft_voting", "metrics": benchmark_result.metrics})
        log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.error(f"Soft voting failed: {e}")

    # 3. Blending (equal weights)
    log.info("\n=== Blending Ensemble (Equal Weights) ===")
    try:
        blending_equal = BlendingEnsemble(base_models, weights=None)
        blending_equal.fit(X_train, y_train)
        benchmark_result = benchmark_model(
            blending_equal,
            X_train,
            X_test,
            y_train,
            y_test,
            filenames_test,
            "BlendingEqual",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "blending_equal", "metrics": benchmark_result.metrics, "weights": blending_equal.weights})
        log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.error(f"Blending equal failed: {e}")

    # 4. Blending (learned weights)
    log.info("\n=== Blending Ensemble (Learned Weights) ===")
    try:
        learned_weights = learn_blending_weights(base_models, X_val, y_val)
        blending_learned = BlendingEnsemble(base_models, weights=learned_weights)
        blending_learned.fit(X_train, y_train)
        benchmark_result = benchmark_model(
            blending_learned,
            X_train,
            X_test,
            y_train,
            y_test,
            filenames_test,
            "BlendingLearned",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append(
            {"method": "blending_learned", "metrics": benchmark_result.metrics, "weights": learned_weights}
        )
        log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
        log.info(f"Weights: {dict(zip(base_names, learned_weights))}")
    except Exception as e:
        log.error(f"Blending learned failed: {e}")

    # 5. Stacking
    log.info("\n=== Stacking Ensemble ===")
    try:
        stacking = StackingEnsemble(base_models)
        stacking.fit(X_train, y_train)
        benchmark_result = benchmark_model(
            stacking,
            X_train,
            X_test,
            y_train,
            y_test,
            filenames_test,
            "Stacking",
            None,
            {},
            None,
            FEATURE_COUNT,
            "FULL" if USE_FULL_FEATURES else "FAST",
        )
        results.append({"method": "stacking", "metrics": benchmark_result.metrics})
        log.info(f"Accuracy: {benchmark_result.metrics['accuracy']:.4f}")
    except Exception as e:
        log.error(f"Stacking failed: {e}")

    # Print comparison
    log.info("\n=== Ensemble Comparison ===")
    log.info(f"{'Method':<20} {'Accuracy':<12} {'F1':<12} {'ROC-AUC':<12}")
    log.info("-" * 60)

    for result in results:
        method = result["method"]
        metrics = result["metrics"]
        accuracy = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        roc_auc = metrics.get("roc_auc", float("nan"))
        roc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
        log.info(f"{method:<20} {accuracy:<12.4f} {f1:<12.4f} {roc_str:<12}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "ensemble_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

