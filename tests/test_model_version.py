"""Tests for model versioning & drift detection."""
from __future__ import annotations

from src.model_version import (
    compute_feature_hash,
    compute_params_hash,
    create_model_metadata,
    validate_model_metadata,
)


def test_compute_feature_hash():
    """Feature hash should be consistent."""
    hash1 = compute_feature_hash(71, 'FAST')
    hash2 = compute_feature_hash(71, 'FAST')
    assert hash1 == hash2

    # Different configs produce different hashes
    hash_full = compute_feature_hash(95, 'FULL')
    assert hash_full != hash1


def test_compute_params_hash():
    """Params hash should ignore irrelevant fields."""
    params1 = {'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42}
    params2 = {'learning_rate': 0.05, 'max_depth': 6, 'random_state': 999}

    # Should be same (random_state excluded)
    hash1 = compute_params_hash(params1)
    hash2 = compute_params_hash(params2)
    assert hash1 == hash2

    # Different params produce different hash
    params3 = {'learning_rate': 0.1, 'max_depth': 6}
    hash3 = compute_params_hash(params3)
    assert hash3 != hash1


def test_create_model_metadata():
    """Metadata should contain all required fields."""
    params = {'learning_rate': 0.05, 'max_depth': 6}
    meta = create_model_metadata(
        feature_count=71,
        feature_mode='FAST',
        params=params,
        n_samples=100,
    )

    assert meta['version'] == 1
    assert meta['feature_count'] == 71
    assert meta['feature_mode'] == 'FAST'
    assert meta['feature_hash'] is not None
    assert meta['params_hash'] is not None
    assert meta['n_samples'] == 100


def test_validate_model_metadata_valid():
    """Valid metadata should pass validation."""
    params = {'learning_rate': 0.05, 'max_depth': 6}
    meta = create_model_metadata(71, 'FAST', params, 100)

    is_valid, mismatches = validate_model_metadata(
        meta,
        current_feature_count=71,
        current_mode='FAST',
        current_params=params,
    )

    assert is_valid is True
    assert len(mismatches) == 0


def test_validate_model_metadata_feature_count_mismatch():
    """Feature count mismatch should be detected."""
    params = {'learning_rate': 0.05, 'max_depth': 6}
    meta = create_model_metadata(71, 'FAST', params, 100)

    is_valid, mismatches = validate_model_metadata(
        meta,
        current_feature_count=95,  # Mismatch!
        current_mode='FAST',
        current_params=params,
    )

    assert is_valid is False
    assert any('Feature count' in m for m in mismatches)


def test_validate_model_metadata_mode_mismatch():
    """Feature mode mismatch should be detected."""
    params = {'learning_rate': 0.05, 'max_depth': 6}
    meta = create_model_metadata(71, 'FAST', params, 100)

    is_valid, mismatches = validate_model_metadata(
        meta,
        current_feature_count=71,
        current_mode='FULL',  # Mismatch!
        current_params=params,
    )

    assert is_valid is False
    assert any('Feature mode' in m for m in mismatches)


def test_validate_model_metadata_params_change():
    """Changed params should warn but not fail."""
    params_old = {'learning_rate': 0.05, 'max_depth': 6}
    meta = create_model_metadata(71, 'FAST', params_old, 100)

    params_new = {'learning_rate': 0.1, 'max_depth': 8}
    is_valid, mismatches = validate_model_metadata(
        meta,
        current_feature_count=71,
        current_mode='FAST',
        current_params=params_new,
    )

    # Params mismatch is a warning, not a hard error
    assert not is_valid
    assert any('Hyperparameters changed' in m for m in mismatches)


def test_validate_model_metadata_invalid_input():
    """Invalid metadata should be handled gracefully."""
    is_valid, mismatches = validate_model_metadata(
        {},  # Empty dict
        current_feature_count=71,
        current_mode='FAST',
        current_params={},
    )

    assert is_valid is False
    assert len(mismatches) > 0

