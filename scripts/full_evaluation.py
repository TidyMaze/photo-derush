#!/usr/bin/env python3
"""Comprehensive model evaluation with per-class metrics, ROC/PR curves, calibration plots, and error analysis.

Usage:
    poetry run python scripts/full_evaluation.py [IMAGE_DIR] [--model-path PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, load_baseline
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.training_core import train_keep_trash_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("full_evaluation")


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    """Compute per-class precision, recall, and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    metrics = {
        "trash": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "keep": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

    return metrics


def compute_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute ROC and PR curve data."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def compute_calibration_data(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration curve data."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    fraction_positives = []
    mean_predicted_values = []
    counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            fraction_positives.append(float(y_true[in_bin].mean()))
            mean_predicted_values.append(float(y_proba[in_bin].mean()))
            counts.append(int(in_bin.sum()))
        else:
            fraction_positives.append(0.0)
            mean_predicted_values.append(float((bin_lower + bin_upper) / 2))
            counts.append(0)

    return {
        "fraction_positives": fraction_positives,
        "mean_predicted_values": mean_predicted_values,
        "counts": counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def analyze_errors(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, filenames: list[str]
) -> dict:
    """Analyze misclassified examples."""
    errors = {
        "false_positives": [],  # Predicted keep, actually trash
        "false_negatives": [],  # Predicted trash, actually keep
    }

    for i, (true_label, pred_label, proba, fname) in enumerate(zip(y_true, y_pred, y_proba, filenames)):
        if true_label != pred_label:
            if pred_label == 1 and true_label == 0:
                errors["false_positives"].append({"filename": fname, "probability": float(proba)})
            elif pred_label == 0 and true_label == 1:
                errors["false_negatives"].append({"filename": fname, "probability": float(proba)})

    # Sort by probability (most confident errors first)
    errors["false_positives"].sort(key=lambda x: x["probability"], reverse=True)
    errors["false_negatives"].sort(key=lambda x: x["probability"])

    return errors


def plot_curves(curves_data: dict, output_dir: Path) -> list[Path]:
    """Generate ROC, PR, and calibration plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("Matplotlib not available, skipping plots")
        return []

    plot_files = []

    # ROC curve
    if "roc" in curves_data:
        plt.figure(figsize=(8, 6))
        plt.plot(curves_data["roc"]["fpr"], curves_data["roc"]["tpr"], label="ROC curve")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        roc_path = output_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=150)
        plt.close()
        plot_files.append(roc_path)

    # PR curve
    if "pr" in curves_data:
        plt.figure(figsize=(8, 6))
        plt.plot(curves_data["pr"]["recall"], curves_data["pr"]["precision"], label="PR curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        pr_path = output_dir / "pr_curve.png"
        plt.savefig(pr_path, dpi=150)
        plt.close()
        plot_files.append(pr_path)

    # Calibration plot
    if "calibration" in curves_data:
        cal_data = curves_data["calibration"]
        plt.figure(figsize=(8, 6))
        plt.plot(cal_data["mean_predicted_values"], cal_data["fraction_positives"], "s-", label="Calibrated")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True)
        cal_path = output_dir / "calibration_plot.png"
        plt.savefig(cal_path, dpi=150)
        plt.close()
        plot_files.append(cal_path)

    return plot_files


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("image_dir", nargs="?", default=None, help="Image directory")
    parser.add_argument("--model-path", default=None, help="Path to existing model (if not provided, trains new model)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots and JSON")
    args = parser.parse_args()

    # Determine image directory
    if args.image_dir:
        image_dir = os.path.expanduser(args.image_dir)
    else:
        image_dir = os.path.expanduser("~/Pictures/photo-dataset")

    if not os.path.isdir(image_dir):
        log.error(f"Image directory does not exist: {image_dir}")
        return 1

    # Output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent / ".cache" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset from {image_dir}")

    # Initialize repository
    repo_path = os.path.join(image_dir, ".ratings_tags.json")
    repo = RatingsTagsRepository(path=repo_path)

    # Build dataset
    X, y, filenames = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 20:
        log.error(f"Insufficient labeled data: {len(y)} samples")
        return 1

    # Fixed stratified train/test split
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, stratify=y, random_state=args.random_state
    )

    log.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train or load model
    if args.model_path and os.path.exists(args.model_path):
        log.info(f"Loading existing model from {args.model_path}")
        import joblib

        data = joblib.load(args.model_path)
        clf = data.get("model") if isinstance(data, dict) else data
    else:
        log.info("Training new model...")
        temp_model_path = os.path.join(image_dir, ".eval_temp_model.joblib")
        result = train_keep_trash_model(
            image_dir=image_dir,
            model_path=temp_model_path,
            repo=repo,
            displayed_filenames=filenames_train,
            n_estimators=200,
            random_state=args.random_state,
        )
        if result is None:
            log.error("Training failed")
            return 1
        import joblib

        data = joblib.load(temp_model_path)
        clf = data["model"]
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

    if y_proba is None:
        log.error("Model does not support probability predictions")
        return 1

    # Comprehensive evaluation
    log.info("\n=== Comprehensive Evaluation ===")

    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(y_test, y_pred, y_proba)
    log.info("\nPer-Class Metrics:")
    for class_name, metrics in per_class_metrics.items():
        log.info(f"  {class_name}:")
        log.info(f"    Precision: {metrics['precision']:.4f}")
        log.info(f"    Recall: {metrics['recall']:.4f}")
        log.info(f"    F1: {metrics['f1']:.4f}")
        log.info(f"    Support: {metrics['support']}")

    # Classification report
    log.info("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["trash", "keep"], zero_division=0)
    log.info(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log.info("\nConfusion Matrix:")
    log.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    log.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # ROC and PR curves
    curves_data = compute_roc_pr_curves(y_test, y_proba)

    # Calibration data
    calibration_data = compute_calibration_data(y_test, y_proba)
    curves_data["calibration"] = calibration_data

    # Error analysis
    error_analysis = analyze_errors(y_test, y_pred, y_proba, filenames_test)
    log.info("\nError Analysis:")
    log.info(f"  False Positives (predicted keep, actually trash): {len(error_analysis['false_positives'])}")
    log.info(f"  False Negatives (predicted trash, actually keep): {len(error_analysis['false_negatives'])}")

    if error_analysis["false_positives"]:
        log.info("\n  Top 5 False Positives (highest probability):")
        for i, err in enumerate(error_analysis["false_positives"][:5]):
            log.info(f"    {i+1}. {err['filename']}: p={err['probability']:.4f}")

    if error_analysis["false_negatives"]:
        log.info("\n  Top 5 False Negatives (lowest probability):")
        for i, err in enumerate(error_analysis["false_negatives"][:5]):
            log.info(f"    {i+1}. {err['filename']}: p={err['probability']:.4f}")

    # Benchmark result
    benchmark_result = benchmark_model(
        clf,
        X_train,
        X_test,
        y_train,
        y_test,
        filenames_test,
        "FullEvaluation",
        args.model_path,
        {},
        None,
        FEATURE_COUNT,
        "FULL" if USE_FULL_FEATURES else "FAST",
    )

    # Compare with baseline
    baseline = load_baseline()
    comparison = None
    if baseline:
        from src.benchmark import compare_results

        comparison = compare_results(benchmark_result, baseline)
        log.info("\n=== Comparison with Baseline ===")
        log.info("Metric improvements:")
        for metric_name, improvement_pct in comparison["improvements"].items():
            delta = comparison["deltas"][metric_name]
            direction = "↑" if improvement_pct > 0 else "↓"
            log.info(f"  {metric_name}: {delta:+.4f} ({improvement_pct:+.2f}%) {direction}")

    # Generate plots
    plot_files = plot_curves(curves_data, output_dir)
    if plot_files:
        log.info(f"\nGenerated {len(plot_files)} plots in {output_dir}")

    # Save comprehensive results
    results = {
        "benchmark": benchmark_result.metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "curves": curves_data,
        "error_analysis": error_analysis,
        "classification_report": report,
        "comparison": comparison,
        "plot_files": [str(p) for p in plot_files],
    }

    results_path = output_dir / "full_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved comprehensive results to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

