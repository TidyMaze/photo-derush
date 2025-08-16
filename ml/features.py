import logging

import cv2
import numpy as np
from pathlib import Path

from .features_cv import compute_quality_features_from_path, FEATURE_NAMES, compute_feature_vector as _cv_compute

# Unified technical quality features (delegates to features_cv). Legacy keys replaced by FEATURE_NAMES.
# extract_features now returns dict with the canonical FEATURE_NAMES set.
# Existing callers using feature_vector continue to receive (np.array, keys) in stable order.

# Technical quality features for an image (OpenCV)
def extract_features(image_path):
    try:
        d = compute_quality_features_from_path(Path(image_path))
    except Exception:  # noqa: PERF203
        logging.exception("[Features] Failed extraction for %s", image_path)
        return None
    return d

def feature_vector(image_path):
    logging.info("[Predict] Extracting feature vector for image=%s (unified cv)", image_path)
    vec, keys = _cv_compute(image_path)
    return vec, keys
