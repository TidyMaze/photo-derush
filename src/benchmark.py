"""Benchmark infrastructure for model evaluation and comparison."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
BASELINE_PATH = CACHE_DIR / "baseline_metrics.json"
BENCHMARK_RESULTS_PATH = CACHE_DIR / "benchmark_results.json"


@dataclass
class BenchmarkResult:
    """Encapsulates benchmark evaluation results."""

    timestamp: str
    model_name: str
    model_path: str | None
    hyperparameters: dict[str, Any]
    dataset_info: dict[str, Any]
    metrics: dict[str, float]
    feature_importances: list[tuple[int, float]] | None
    test_filenames: list[str]
    train_size: int
    test_size: int
    feature_count: int
    feature_mode: str  # "FAST" or "FULL"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
        except ValueError:
            # Can happen if only one class present
            metrics["roc_auc"] = float("nan")
            metrics["brier_score"] = float("nan")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["confusion_tn"] = int(tn)
    metrics["confusion_fp"] = int(fp)
    metrics["confusion_fn"] = int(fn)
    metrics["confusion_tp"] = int(tp)

    return metrics


def benchmark_model(
    clf: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    test_filenames: list[str],
    model_name: str = "XGBoost",
    model_path: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    feature_importances: list[tuple[int, float]] | None = None,
    feature_count: int = 71,
    feature_mode: str = "FAST",
) -> BenchmarkResult:
    """Train and evaluate a model with fixed train/test split.

    Args:
        clf: Fitted classifier with predict() and predict_proba() methods
        X_train: Training features
        X_test: Test features
        y_train: Training labels (0=trash, 1=keep)
        y_test: Test labels (0=trash, 1=keep)
        test_filenames: Filenames corresponding to test set
        model_name: Name of the model
        model_path: Path to saved model file (optional)
        hyperparameters: Model hyperparameters dict
        feature_importances: List of (feature_idx, importance) tuples
        feature_count: Number of features used
        feature_mode: "FAST" or "FULL"

    Returns:
        BenchmarkResult with all metrics and metadata
    """
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception as e:
            logging.warning(f"[benchmark] predict_proba failed: {e}")

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Dataset info
    n_train_keep = int(np.sum(y_train == 1))
    n_train_trash = int(np.sum(y_train == 0))
    n_test_keep = int(np.sum(y_test == 1))
    n_test_trash = int(np.sum(y_test == 0))

    dataset_info = {
        "train_size": len(y_train),
        "train_keep": n_train_keep,
        "train_trash": n_train_trash,
        "train_balance_ratio": max(n_train_keep, n_train_trash) / min(n_train_keep, n_train_trash) if min(n_train_keep, n_train_trash) > 0 else 0.0,
        "test_size": len(y_test),
        "test_keep": n_test_keep,
        "test_trash": n_test_trash,
        "test_balance_ratio": max(n_test_keep, n_test_trash) / min(n_test_keep, n_test_trash) if min(n_test_keep, n_test_trash) > 0 else 0.0,
    }

    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model_name=model_name,
        model_path=model_path,
        hyperparameters=hyperparameters or {},
        dataset_info=dataset_info,
        metrics=metrics,
        feature_importances=feature_importances,
        test_filenames=test_filenames,
        train_size=len(y_train),
        test_size=len(y_test),
        feature_count=feature_count,
        feature_mode=feature_mode,
    )

    return result


def save_benchmark(result: BenchmarkResult, path: Path | None = None) -> None:
    """Save benchmark result to JSON file."""
    if path is None:
        path = BENCHMARK_RESULTS_PATH

    # Convert to dict, handling numpy types
    data = asdict(result)

    # Load existing results if file exists
    all_results = []
    if path.exists():
        try:
            with open(path) as f:
                all_results = json.load(f)
        except Exception as e:
            logging.warning(f"[benchmark] Failed to load existing results: {e}")

    # Append new result
    all_results.append(data)

    # Save
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logging.info(f"[benchmark] Saved result to {path}")


def load_baseline(path: Path | None = None) -> BenchmarkResult | None:
    """Load baseline metrics from file."""
    if path is None:
        path = BASELINE_PATH

    if not path.exists():
        logging.info(f"[benchmark] No baseline found at {path}")
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        return BenchmarkResult(**data)
    except Exception as e:
        logging.warning(f"[benchmark] Failed to load baseline: {e}")
        return None


def save_baseline(result: BenchmarkResult, path: Path | None = None) -> None:
    """Save baseline metrics to file."""
    if path is None:
        path = BASELINE_PATH

    data = asdict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logging.info(f"[benchmark] Saved baseline to {path}")


def compare_results(new_result: BenchmarkResult, baseline: BenchmarkResult) -> dict[str, Any]:
    """Compare new benchmark result against baseline.

    Returns:
        Dict with metric deltas and improvement percentages
    """
    comparison = {
        "baseline": {
            "model_name": baseline.model_name,
            "timestamp": baseline.timestamp,
            "metrics": baseline.metrics,
        },
        "new": {
            "model_name": new_result.model_name,
            "timestamp": new_result.timestamp,
            "metrics": new_result.metrics,
        },
        "deltas": {},
        "improvements": {},
    }

    # Compute deltas for each metric
    for metric_name in new_result.metrics:
        if metric_name in baseline.metrics:
            baseline_val = baseline.metrics[metric_name]
            new_val = new_result.metrics[metric_name]

            # Skip NaN values
            if np.isnan(baseline_val) or np.isnan(new_val):
                continue

            delta = new_val - baseline_val
            comparison["deltas"][metric_name] = float(delta)

            # Improvement percentage (positive = better)
            if metric_name == "brier_score":
                # Lower is better for Brier score
                if baseline_val > 0:
                    improvement_pct = -100 * delta / baseline_val
                else:
                    improvement_pct = 0.0
            else:
                # Higher is better for other metrics
                if baseline_val > 0:
                    improvement_pct = 100 * delta / baseline_val
                else:
                    improvement_pct = 0.0

            comparison["improvements"][metric_name] = float(improvement_pct)

    return comparison


__all__ = [
    "BenchmarkResult",
    "benchmark_model",
    "save_benchmark",
    "load_baseline",
    "save_baseline",
    "compare_results",
    "compute_metrics",
    "BASELINE_PATH",
    "BENCHMARK_RESULTS_PATH",
]

