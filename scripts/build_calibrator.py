#!/usr/bin/env python3
"""Build a calibrator for an existing saved model using labeled images.

Usage:
  poetry run python scripts/build_calibrator.py [IMAGE_DIR] [MODEL_PATH]

This will load the model at MODEL_PATH (defaults to /tmp/photo_final.joblib),
extract features from IMAGE_DIR, split off a small calibration set, and fit
CalibratedClassifierCV(cv='prefit', method='sigmoid') using the model.
The calibrator will be saved as MODEL_PATH + '.calib.joblib'.
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("build_calibrator")

DEFAULT_DIR = os.path.expanduser("~/Pictures/photo-dataset")
DEFAULT_MODEL = "/tmp/photo_final.joblib"


def main():
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.calibration import CalibratedClassifierCV
        import joblib
    except Exception as e:
        log.exception("Missing dependencies: %s", e)
        sys.exit(2)

    image_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    image_dir = os.path.expanduser(image_dir)
    model_path = os.path.expanduser(model_path)

    if not os.path.isdir(image_dir):
        log.error("Image directory does not exist: %s", image_dir)
        sys.exit(2)
    if not os.path.isfile(model_path):
        log.error("Model file does not exist: %s", model_path)
        sys.exit(2)

    try:
        from src.dataset import build_dataset
        from src.inference import load_model
        from src.features import FEATURE_COUNT
    except Exception as e:
        log.exception("Import error: %s", e)
        sys.exit(2)

    log.info("Loading dataset from %s", image_dir)
    # Create scoped repo like other scripts do
    try:
        from src.model import RatingsTagsRepository
        repo_path = os.path.join(image_dir, '.ratings_tags.json')
        repo = RatingsTagsRepository(path=repo_path)
    except Exception:
        repo = None

    X, y, _ = build_dataset(image_dir, repo)
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) < 20:
        log.error("Not enough labeled samples to build calibrator: %d", len(y))
        sys.exit(1)

    # Split into train+calib/test; we only need calib to fit calibrator.
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=42)

    loaded = load_model(model_path)
    if loaded is None:
        log.error("Failed to load model %s", model_path)
        sys.exit(1)

    model, meta, existing_calibrator = loaded.model, loaded.meta, loaded.calibrator

    if existing_calibrator is not None:
        log.info("Model already has a calibrator; exiting")
        sys.exit(0)

    # We need to fit calibrator on the features in the same space expected by the model.
    # If model is a Pipeline (scaler + xgb), extract scaler to transform X_calib first.
    scaler = None
    try:
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
    except Exception:
        scaler = None

    if scaler is not None:
        X_calib_trans = scaler.transform(X_calib)
    else:
        X_calib_trans = X_calib

    try:
        calibrator = CalibratedClassifierCV(model.named_steps['xgb'] if hasattr(model, 'named_steps') else model, cv='prefit', method='sigmoid')
        calibrator.fit(X_calib_trans, y_calib)
    except Exception as e:
        log.exception("Failed to fit calibrator: %s", e)
        sys.exit(1)

    calib_path = model_path + '.calib.joblib'
    try:
        joblib.dump(calibrator, calib_path)
        log.info("Saved calibrator to %s", calib_path)
    except Exception as e:
        log.exception("Failed to save calibrator: %s", e)
        sys.exit(1)

    # Optionally print test metrics
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        if scaler is not None:
            X_test_trans = scaler.transform(X_test)
        else:
            X_test_trans = X_test
        y_proba_uncal = (model.predict_proba(X_test_trans)[:, 1] if hasattr(model, 'predict_proba') else None)
        y_proba_cal = calibrator.predict_proba(X_test_trans)[:, 1]
        if y_proba_uncal is not None:
            print('uncal ROC AUC:', roc_auc_score(y_test, y_proba_uncal))
        print('cal ROC AUC:', roc_auc_score(y_test, y_proba_cal))
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
