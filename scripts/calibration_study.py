#!/usr/bin/env python3
"""Study probability calibration methods.

Usage:
    poetry run python scripts/calibration_study.py [IMAGE_DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import benchmark_model, save_benchmark
from src.dataset import build_dataset
from src.features import FEATURE_COUNT, USE_FULL_FEATURES
from src.model import RatingsTagsRepository
from src.tuning import load_best_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("calibration")


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def temperature_scaling(y_proba: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    """Apply temperature scaling (learn optimal temperature parameter)."""
    from scipy.optimize import minimize

    def objective(temp):
        scaled_proba = y_proba ** (1.0 / temp)
        scaled_proba = np.clip(scaled_proba, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y_true * np.log(scaled_proba) + (1 - y_true) * np.log(1 - scaled_proba))
        return log_loss

    result = minimize(objective, 1.0, method="BFGS", bounds=[(0.1, 10.0)])
    temp = result.x[0] if result.success else 1.0
    scaled_proba = y_proba ** (1.0 / temp)
    return float(temp), scaled_proba


def main():
    parser = argparse.ArgumentParser(description="Calibration study")
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

    # Split: train (60%), calib (20%), test (20%)
    X_train, X_rest, y_train, y_rest, filenames_train, filenames_rest = train_test_split(
        X, y, filenames, test_size=0.4, stratify=y, random_state=args.random_state
    )
    X_calib, X_test, y_calib, y_test, filenames_calib, filenames_test = train_test_split(
        X_rest, y_rest, filenames_rest, test_size=0.5, stratify=y_rest, random_state=args.random_state
    )

    log.info(f"Train: {len(y_train)}, Calib: {len(y_calib)}, Test: {len(y_test)}")

    # Train base model
    xgb_params = load_best_params() or {}
    xgb_params = {k: v for k, v in xgb_params.items() if not k.startswith("_")}

    n_keep = int(np.sum(y_train == 1))
    n_trash = int(np.sum(y_train == 0))
    scale_pos_weight = n_trash / n_keep if n_keep > 0 else 1.0

    base_clf = Pipeline(
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
    )

    log.info("Training base model...")
    base_clf.fit(X_train, y_train)

    # Get uncalibrated predictions
    y_proba_uncal = base_clf.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, y_proba_uncal)
    ece_uncal = compute_ece(y_test, y_proba_uncal)

    results = [
        {
            "method": "uncalibrated",
            "brier_score": float(brier_uncal),
            "ece": float(ece_uncal),
            "metrics": benchmark_model(
                base_clf, X_train, X_test, y_train, y_test, filenames_test, "Uncalibrated", None, {}, None, FEATURE_COUNT, "FULL" if USE_FULL_FEATURES else "FAST"
            ).metrics,
        }
    ]

    # 1. Platt Scaling (Sigmoid)
    log.info("\n=== Platt Scaling (Sigmoid) ===")
    try:
        calib_platt = CalibratedClassifierCV(base_clf.named_steps["xgb"], cv="prefit", method="sigmoid")
        calib_platt.fit(X_calib, y_calib)
        y_proba_platt = calib_platt.predict_proba(X_test)[:, 1]
        brier_platt = brier_score_loss(y_test, y_proba_platt)
        ece_platt = compute_ece(y_test, y_proba_platt)

        # Create wrapper for benchmark
        class CalibratedWrapper:
            def __init__(self, base, calibrator):
                self.base = base
                self.calibrator = calibrator

            def predict(self, X):
                return self.calibrator.predict(X)

            def predict_proba(self, X):
                return self.calibrator.predict_proba(X)

        wrapper = CalibratedWrapper(base_clf, calib_platt)
        benchmark_result = benchmark_model(
            wrapper, X_train, X_test, y_train, y_test, filenames_test, "PlattScaling", None, {}, None, FEATURE_COUNT, "FULL" if USE_FULL_FEATURES else "FAST"
        )

        results.append(
            {
                "method": "platt_scaling",
                "brier_score": float(brier_platt),
                "ece": float(ece_platt),
                "metrics": benchmark_result.metrics,
            }
        )
        log.info(f"Brier score: {brier_platt:.4f} (improvement: {brier_uncal - brier_platt:+.4f})")
        log.info(f"ECE: {ece_platt:.4f} (improvement: {ece_uncal - ece_platt:+.4f})")
    except Exception as e:
        log.error(f"Platt scaling failed: {e}")

    # 2. Isotonic Regression
    log.info("\n=== Isotonic Regression ===")
    try:
        calib_isotonic = CalibratedClassifierCV(base_clf.named_steps["xgb"], cv="prefit", method="isotonic")
        calib_isotonic.fit(X_calib, y_calib)
        y_proba_isotonic = calib_isotonic.predict_proba(X_test)[:, 1]
        brier_isotonic = brier_score_loss(y_test, y_proba_isotonic)
        ece_isotonic = compute_ece(y_test, y_proba_isotonic)

        wrapper = CalibratedWrapper(base_clf, calib_isotonic)
        benchmark_result = benchmark_model(
            wrapper, X_train, X_test, y_train, y_test, filenames_test, "IsotonicRegression", None, {}, None, FEATURE_COUNT, "FULL" if USE_FULL_FEATURES else "FAST"
        )

        results.append(
            {
                "method": "isotonic_regression",
                "brier_score": float(brier_isotonic),
                "ece": float(ece_isotonic),
                "metrics": benchmark_result.metrics,
            }
        )
        log.info(f"Brier score: {brier_isotonic:.4f} (improvement: {brier_uncal - brier_isotonic:+.4f})")
        log.info(f"ECE: {ece_isotonic:.4f} (improvement: {ece_uncal - ece_isotonic:+.4f})")
    except Exception as e:
        log.error(f"Isotonic regression failed: {e}")

    # 3. Temperature Scaling
    log.info("\n=== Temperature Scaling ===")
    try:
        y_proba_calib = base_clf.predict_proba(X_calib)[:, 1]
        temp, y_proba_temp = temperature_scaling(y_proba_calib, y_calib)

        # Apply to test set
        y_proba_test_base = base_clf.predict_proba(X_test)[:, 1]
        _, y_proba_temp_test = temperature_scaling(y_proba_test_base, y_test)

        brier_temp = brier_score_loss(y_test, y_proba_temp_test)
        ece_temp = compute_ece(y_test, y_proba_temp_test)

        class TempScaledWrapper:
            def __init__(self, base, temp_param):
                self.base = base
                self.temp = temp_param

            def predict(self, X):
                proba = self.predict_proba(X)[:, 1]
                return (proba >= 0.5).astype(int)

            def predict_proba(self, X):
                proba_base = self.base.predict_proba(X)[:, 1]
                proba_scaled = proba_base ** (1.0 / self.temp)
                return np.column_stack([1 - proba_scaled, proba_scaled])

        wrapper = TempScaledWrapper(base_clf, temp)
        benchmark_result = benchmark_model(
            wrapper, X_train, X_test, y_train, y_test, filenames_test, "TemperatureScaling", None, {}, None, FEATURE_COUNT, "FULL" if USE_FULL_FEATURES else "FAST"
        )

        results.append(
            {
                "method": "temperature_scaling",
                "temperature": float(temp),
                "brier_score": float(brier_temp),
                "ece": float(ece_temp),
                "metrics": benchmark_result.metrics,
            }
        )
        log.info(f"Temperature: {temp:.4f}")
        log.info(f"Brier score: {brier_temp:.4f} (improvement: {brier_uncal - brier_temp:+.4f})")
        log.info(f"ECE: {ece_temp:.4f} (improvement: {ece_uncal - ece_temp:+.4f})")
    except Exception as e:
        log.error(f"Temperature scaling failed: {e}")

    # Print comparison
    log.info("\n=== Calibration Comparison ===")
    log.info(f"{'Method':<25} {'Brier Score':<15} {'ECE':<15} {'Improvement':<15}")
    log.info("-" * 70)

    for result in results:
        method = result["method"]
        brier = result.get("brier_score", 0.0)
        ece = result.get("ece", 0.0)
        improvement = brier_uncal - brier if method != "uncalibrated" else 0.0
        improvement_str = f"{improvement:+.4f}" if method != "uncalibrated" else "baseline"
        log.info(f"{method:<25} {brier:<15.4f} {ece:<15.4f} {improvement_str:<15}")

    # Save results
    output_path = args.out or (Path(__file__).resolve().parent.parent / ".cache" / "calibration_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nSaved results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

