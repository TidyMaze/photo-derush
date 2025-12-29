#!/usr/bin/env python3
"""Holdout evaluation + calibration script.

Splits dataset into train/calib/test (70/10/20), trains XGBoost with tuned params if available,
reports uncalibrated vs calibrated metrics on the test set.

Usage:
  poetry run python scripts/holdout_calibrate.py [IMAGE_DIR]
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("holdout_calibrate")

DEFAULT_DIR = os.path.expanduser("~/Pictures/photo-dataset")


def main():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    try:
        from src.dataset import build_dataset
        from src.tuning import load_best_params
    except Exception as e:
        log.exception("Import error: %s", e)
        sys.exit(2)

    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    image_dir = os.path.expanduser(image_dir)
    if not os.path.isdir(image_dir):
        log.error("Image directory does not exist: %s", image_dir)
        sys.exit(2)

    log.info("Loading dataset from %s", image_dir)
    # Create scoped repository like other scripts do so build_dataset can query image state
    try:
        from src.model import RatingsTagsRepository
        repo_path = os.path.join(image_dir, '.ratings_tags.json')
        repo = RatingsTagsRepository(path=repo_path)
        log.info("Scoped repo: %s", repo_path)
    except Exception:
        repo = None
    X, y, _ = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 20:
        log.error("Not enough samples for holdout evaluation: %d", len(y))
        sys.exit(1)

    # splits: train (70%), calib (10%), test (20%)
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.6667, stratify=y_rest, random_state=42)
    # ~70/10/20 split

    log.info("Splits: train=%d, calib=%d, test=%d", len(y_train), len(y_calib), len(y_test))

    # Load tuned params if any
    tuned = load_best_params()
    xgb_params = {}
    if tuned:
        xgb_params = {k: v for k, v in tuned.items() if not k.startswith('_')}
        log.info("Using tuned params: %s", xgb_params)
    else:
        log.info("No tuned params found; using defaults")

    import xgboost as xgb
    from sklearn.pipeline import make_pipeline

    # compute scale_pos_weight based on train
    n_keep = int((y_train == 1).sum())
    n_trash = int((y_train == 0).sum())
    scale_pos_weight = (n_trash / n_keep) if n_keep > 0 else 1.0
    log.info('Train class balance: keep=%d trash=%d scale_pos_weight=%.3f', n_keep, n_trash, scale_pos_weight)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBClassifier(random_state=42, n_jobs=-1, scale_pos_weight=scale_pos_weight, objective='binary:logistic', eval_metric='logloss', **xgb_params))
    ])

    log.info("Training base classifier on train set...")
    clf.fit(X_train, y_train)

    # Evaluate uncalibrated
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_proba):
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion': confusion_matrix(y_true, y_pred).tolist(),
        }

    uncal = metrics(y_test, y_pred, y_proba)

    # Calibrate using calib set
    log.info("Calibrating probabilities using calib set (cv='prefit')...")
    calibrator = CalibratedClassifierCV(clf.named_steps['xgb'], cv='prefit', method='sigmoid')
    # Need to fit scaler + raw xgb, so transform calib features
    X_calib_scaled = clf.named_steps['scaler'].transform(X_calib)
    calibrator.fit(X_calib_scaled, y_calib)

    # For calibrated predictions, pass scaled X_test to calibrator
    X_test_scaled = clf.named_steps['scaler'].transform(X_test)
    y_proba_cal = calibrator.predict_proba(X_test_scaled)[:, 1]
    # For class predictions after calibration, threshold 0.5
    y_pred_cal = (y_proba_cal >= 0.5).astype(int)
    cal = metrics(y_test, y_pred_cal, y_proba_cal)

    print('\nUNCALIBRATED METRICS:\n')
    for k, v in uncal.items():
        print(f"{k}: {v}")

    print('\nCALIBRATED METRICS:\n')
    for k, v in cal.items():
        print(f"{k}: {v}")

    # Save calibrator model and base model to /tmp for inspection
    try:
        import joblib
        joblib.dump(clf, '/tmp/photo_base_pipeline.joblib')
        joblib.dump(calibrator, '/tmp/photo_calibrator.joblib')
        log.info('Saved base pipeline and calibrator to /tmp')
    except Exception as e:
        log.warning('Could not save models: %s', e)

    return 0


if __name__ == '__main__':
    sys.exit(main())
