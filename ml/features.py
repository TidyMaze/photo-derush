import logging

import cv2
import numpy as np
from pathlib import Path

from .features_cv import compute_quality_features_from_path, FEATURE_NAMES, compute_feature_vector as _cv_compute
from .exif_features import extract_exif_features, EXIF_FEATURE_KEYS  # new import

# Unified technical + EXIF features (numeric pixel features stay in FEATURE_NAMES order)

# Technical + EXIF features for an image
def extract_features(image_path):
    try:
        base = compute_quality_features_from_path(Path(image_path))
    except Exception:  # noqa: PERF203
        logging.exception("[Features] Failed CV extraction for %s", image_path)
        base = {k: 0.0 for k in FEATURE_NAMES}
    try:
        exif = extract_exif_features(Path(image_path))
    except Exception:  # noqa: PERF203
        logging.exception("[Features] Failed EXIF extraction for %s", image_path)
        exif = {k: ("unknown" if isinstance(v, str) else 0.0) for k, v in extract_exif_features.__annotations__.items()}  # fallback minimal
    # Merge dictionaries (EXIF keys won't clash with base except by intentional naming)
    merged = {**base, **exif}
    return merged

def all_feature_names(include_strings=True):
    """Return list of all feature keys (pixel + EXIF). If include_strings=False, filter non-numeric EXIF values."""
    if include_strings:
        return FEATURE_NAMES + [k for k in EXIF_FEATURE_KEYS]
    # crude filter: remove obvious string categorical keys
    string_like = {'make','model','lens_make','lens_model','software'}
    return FEATURE_NAMES + [k for k in EXIF_FEATURE_KEYS if k not in string_like]

def feature_vector(image_path, include_strings=False):
    logging.info("[Predict] Extracting feature vector for image=%s (unified cv+exif) include_strings=%s", image_path, include_strings)
    # Get technical features
    base = compute_quality_features_from_path(Path(image_path))
    # Get EXIF features
    try:
        exif = extract_exif_features(Path(image_path))
    except Exception:
        logging.exception("[Features] Failed EXIF extraction for %s", image_path)
        exif = {k: ("unknown" if isinstance(v, str) else 0.0) for k, v in EXIF_FEATURE_KEYS.items()} if hasattr(EXIF_FEATURE_KEYS, 'items') else {}
    # Merge
    merged = {**base, **exif}
    # Determine feature order
    keys = all_feature_names(include_strings=include_strings)
    # Filter and build vector
    vec = []
    for k in keys:
        v = merged.get(k, 0.0)
        if not include_strings and isinstance(v, str):
            v = 0.0
        vec.append(v)

    logging.info("[Predict] Extracted %d features for image=%s: base=%d, exif=%d", len(vec), image_path, len(base), len(exif))
    logging.info("[Predict] Feature keys: %s", keys[:10])  # log first 10 keys for brevity

    return np.array(vec, dtype=np.float64), keys
