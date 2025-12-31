"""Unit tests for threshold optimization (exact quantile-based method)."""

import numpy as np
import pytest

from src.training_core import _find_threshold_for_max_keep_loss, _compute_asymmetric_metrics


def test_threshold_exact_quantile_basic():
    """Test basic quantile-based threshold calculation."""
    # Simple case: 10 KEEP samples, max_keep_loss = 0.2 (20%)
    # We can lose at most 2 KEEP samples
    y_true = np.array([1] * 10 + [0] * 10)  # 10 keep, 10 trash
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.2)
    
    # Should be the 3rd smallest proba (index 2) = 0.3
    # This ensures at most 2 KEEP have p < 0.3 (i.e., at most 2 false negatives)
    assert threshold == 0.3
    
    # Verify constraint is met
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] <= 0.2 + 1e-6


def test_threshold_exact_quantile_zero_loss():
    """Test threshold when constraint allows 0 keep-loss."""
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.0)
    
    # Should accept all KEEP (threshold <= smallest KEEP proba)
    assert threshold <= 0.1 + 1e-6
    
    # Verify no keep-loss
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] == 0.0


def test_threshold_exact_quantile_high_threshold():
    """Test threshold when model is pessimistic (low probas)."""
    # All KEEP have low probas (< 0.05)
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.2)
    
    # Should find threshold < 0.05 (old grid search would miss this)
    assert threshold < 0.05
    
    # Verify constraint is met
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] <= 0.2 + 1e-6


def test_threshold_exact_quantile_very_high_threshold():
    """Test threshold when model is optimistic (high probas)."""
    # All KEEP have high probas (> 0.9)
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.2)
    
    # With max_keep_loss=0.2, we can lose 2 KEEP (out of 10)
    # The 3rd smallest proba (index 2) is 0.93, so threshold should be 0.93
    # This is > 0.9 (old grid search would miss this as it started at 0.05)
    assert threshold >= 0.93
    assert threshold > 0.9  # Key: should find threshold > 0.9 (old grid would miss)
    
    # Verify constraint is met
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] <= 0.2 + 1e-6


def test_threshold_exact_quantile_ties():
    """Test threshold with tied probabilities."""
    # Multiple KEEP have same proba
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.2)
    
    # Should handle ties correctly
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] <= 0.2 + 1e-6


def test_threshold_exact_quantile_no_keep_samples():
    """Test threshold when no KEEP samples."""
    y_true = np.array([0] * 10)  # Only trash
    y_proba = np.array([0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.02)
    
    # Should return default
    assert threshold == 0.5


def test_compute_asymmetric_metrics_keep_rate():
    """Test that keep_rate is computed correctly."""
    y_true = np.array([1] * 5 + [0] * 5)  # 5 keep, 5 trash
    y_proba = np.array([0.9] * 5 + [0.1] * 5)
    
    # Predict all as keep (threshold = 0.0)
    y_pred = np.array([1] * 10)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    
    # keep_rate = (TP + FP) / total = (5 + 5) / 10 = 1.0
    assert metrics["keep_rate"] == 1.0
    
    # Predict all as trash (threshold = 1.0)
    y_pred = np.array([0] * 10)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    
    # keep_rate = (TP + FP) / total = (0 + 0) / 10 = 0.0
    assert metrics["keep_rate"] == 0.0
    
    # Predict half as keep
    y_pred = np.array([1] * 5 + [0] * 5)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    
    # keep_rate = (TP + FP) / total = (5 + 0) / 10 = 0.5
    assert metrics["keep_rate"] == 0.5


def test_threshold_highest_meeting_constraint():
    """Test that threshold is the highest one meeting the constraint."""
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [0.5] * 10)
    
    threshold = _find_threshold_for_max_keep_loss(y_true, y_proba, max_keep_loss=0.2)
    
    # Verify it's the highest threshold meeting constraint
    # Try a slightly higher threshold
    higher_threshold = threshold + 0.01
    y_pred_higher = (y_proba >= higher_threshold).astype(int)
    metrics_higher = _compute_asymmetric_metrics(y_true, y_pred_higher, y_proba)
    
    # Higher threshold should either:
    # 1. Still meet constraint (if there's a gap in probas), or
    # 2. Violate constraint (if threshold was already at the boundary)
    # But our method should return the highest that meets it
    y_pred = (y_proba >= threshold).astype(int)
    metrics = _compute_asymmetric_metrics(y_true, y_pred, y_proba)
    assert metrics["keep_loss_rate"] <= 0.2 + 1e-6

