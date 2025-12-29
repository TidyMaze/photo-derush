"""Tests for balanced training & class-weight handling."""
from __future__ import annotations

import pytest

from src.training_core import _compute_class_balance


def test_compute_class_balance_perfect():
    """Perfect 50-50 split should be flagged as balanced."""
    metrics = _compute_class_balance(50, 50)
    assert metrics['ratio'] == 1.0
    assert metrics['keep_pct'] == 50.0
    assert metrics['trash_pct'] == 50.0
    assert metrics['is_severe'] is False
    assert metrics['recommendation'] == 'Balanced'


def test_compute_class_balance_moderate():
    """2:1 ratio should be flagged as moderate."""
    metrics = _compute_class_balance(60, 30)
    assert metrics['ratio'] == 2.0
    assert metrics['keep_pct'] == pytest.approx(66.67, abs=0.1)
    assert metrics['trash_pct'] == pytest.approx(33.33, abs=0.1)
    assert metrics['is_severe'] is False
    assert metrics['recommendation'] in ['Balanced', 'Moderate']


def test_compute_class_balance_severe():
    """4:1 ratio should be flagged as severe."""
    metrics = _compute_class_balance(80, 20)
    assert metrics['ratio'] == 4.0
    assert metrics['keep_pct'] == 80.0
    assert metrics['trash_pct'] == 20.0
    assert metrics['is_severe'] is True
    assert metrics['recommendation'] == 'Severe'


def test_compute_class_balance_extreme():
    """10:1 ratio should be flagged as severe."""
    metrics = _compute_class_balance(100, 10)
    assert metrics['ratio'] == 10.0
    assert metrics['is_severe'] is True


def test_compute_class_balance_empty():
    """Empty dataset should handle gracefully."""
    metrics = _compute_class_balance(0, 0)
    assert metrics['ratio'] == 0.0
    assert metrics['is_severe'] is False
    assert 'Insufficient' in metrics['recommendation']


def test_compute_class_balance_realistic_moderate():
    """Realistic moderate imbalance (3:2 ratio from real dataset)."""
    # Example: 254 keep, 134 trash (from tuning results)
    metrics = _compute_class_balance(254, 134)
    assert metrics['ratio'] == pytest.approx(254 / 134, rel=0.01)
    assert metrics['is_severe'] is False
    assert metrics['recommendation'] in ['Balanced', 'Moderate']


def test_class_balance_metrics_consistency():
    """Metrics should be consistent regardless of order."""
    metrics1 = _compute_class_balance(40, 160)  # 4:1 reverse
    metrics2 = _compute_class_balance(160, 40)  # 4:1 forward

    # Ratios should be same
    assert metrics1['ratio'] == metrics2['ratio']
    # Both should be severe
    assert metrics1['is_severe'] == metrics2['is_severe'] == True

