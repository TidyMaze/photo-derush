"""Model versioning & drift detection utilities.

Detects mismatches between training environment (features, params) and inference time.
Prevents stale model usage with changed features or hyperparameters.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_feature_hash(feature_count: int, mode: str = "FAST") -> str:
    """Compute stable hash of feature extraction config.

    Used to detect if model was trained with different feature extraction settings.
    """
    config = {"feature_count": feature_count, "mode": mode}
    data = json.dumps(config, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def compute_params_hash(params: dict[str, Any]) -> str:
    """Compute stable hash of hyperparameters.

    Detects if model was trained with different hyperparameters.
    """
    # Exclude random_state and other deterministic params that don't affect predictions
    exclude_keys = {"random_state", "n_jobs", "verbose"}
    filtered = {k: v for k, v in params.items() if k not in exclude_keys}
    data = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def create_model_metadata(feature_count: int, feature_mode: str, params: dict, n_samples: int) -> dict:
    """Create metadata header for model file.

    Stores:
    - Feature hash (detect feature extraction changes)
    - Params hash (detect hyperparameter changes)
    - Training samples count
    - Version (for future schema changes)
    """
    return {
        "version": 1,
        "feature_count": feature_count,
        "feature_mode": feature_mode,
        "feature_hash": compute_feature_hash(feature_count, feature_mode),
        "params_hash": compute_params_hash(params),
        "n_samples": n_samples,
    }


def validate_model_metadata(
    stored_metadata: dict, current_feature_count: int, current_mode: str, current_params: dict
) -> tuple[bool, list[str]]:
    """Check if model metadata matches current environment.

    Returns: (is_valid, [list of mismatch reasons])
    """
    mismatches = []

    if not isinstance(stored_metadata, dict):
        return False, ["Invalid metadata format"]

    # Check version compatibility (v1 only for now)
    if stored_metadata.get("version") != 1:
        mismatches.append(f"Version mismatch: {stored_metadata.get('version')} != 1")

    # Check feature count
    stored_count = stored_metadata.get("feature_count")
    if stored_count != current_feature_count:
        mismatches.append(f"Feature count: {stored_count} → {current_feature_count} (retrain recommended)")

    # Check feature mode
    stored_mode = stored_metadata.get("feature_mode")
    if stored_mode != current_mode:
        mismatches.append(f"Feature mode: {stored_mode} → {current_mode}")

    # Check params hash (warn, don't fail)
    stored_params_hash = stored_metadata.get("params_hash")
    current_hash = compute_params_hash(current_params)
    if stored_params_hash != current_hash:
        mismatches.append(f"Hyperparameters changed (hash: {stored_params_hash} → {current_hash})")

    is_valid = len(mismatches) == 0
    return is_valid, mismatches


__all__ = ["compute_feature_hash", "compute_params_hash", "create_model_metadata", "validate_model_metadata"]
