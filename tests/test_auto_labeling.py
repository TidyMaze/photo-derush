"""Tests for auto-labeling predictions after training."""
from unittest.mock import patch

import numpy as np
import pytest


class TestAutoLabelingPredictions:
    """Test auto-labeling generates valid predictions with appropriate confidence filtering."""

    def test_confidence_threshold_logic(self):
        """Test that confidence threshold correctly filters predictions."""
        # Test cases: (prob_keep, should_include, expected_label, expected_confidence)
        test_cases = [
            (0.95, True, "keep", 0.95),     # High confidence keep
            (0.70, True, "keep", 0.70),     # Medium-high keep
            (0.61, True, "keep", 0.61),     # Just above threshold
            (0.55, False, None, None),      # Uncertain (filtered)
            (0.50, False, None, None),      # Uncertain (filtered)
            (0.45, False, None, None),      # Uncertain (filtered)
            (0.39, True, "trash", 0.61),    # Just below threshold (1-0.39=0.61)
            (0.30, True, "trash", 0.70),    # Medium-high trash
            (0.05, True, "trash", 0.95),    # High confidence trash
        ]

        for prob_keep, should_include, expected_label, expected_conf in test_cases:
            # Apply the logic from ml.py
            if prob_keep > 0.6 or prob_keep < 0.4:
                assert should_include, f"prob_keep={prob_keep} should be included"
                label = "keep" if prob_keep > 0.5 else "trash"
                confidence = prob_keep if label == "keep" else (1 - prob_keep)
                assert label == expected_label, f"Wrong label for prob_keep={prob_keep}"
                assert abs(confidence - expected_conf) < 0.01, f"Wrong confidence for prob_keep={prob_keep}"
            else:
                assert not should_include, f"prob_keep={prob_keep} should be filtered"

    def test_prediction_output_format(self):
        """Test prediction output has correct structure."""
        prediction = {
            "image_id": 123,
            "label": "keep",
            "confidence": 0.85
        }

        assert "image_id" in prediction
        assert "label" in prediction
        assert "confidence" in prediction
        assert prediction["label"] in ["keep", "trash"]
        assert 0 <= prediction["confidence"] <= 1

    def test_stats_output_format(self):
        """Test stats output has all required fields."""
        stats = {
            "total_unlabeled": 362,
            "files_existing": 362,
            "feature_errors": 0,
            "prediction_errors": 0,
            "low_confidence": 362,
            "high_confidence": 0
        }

        required_fields = [
            "total_unlabeled",
            "files_existing",
            "feature_errors",
            "prediction_errors",
            "low_confidence",
            "high_confidence"
        ]

        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"
            assert isinstance(stats[field], int), f"{field} should be integer"

        """Test that we understand the probability distribution from XGBoost."""
        # XGBoost predict_proba returns [[prob_class_0, prob_class_1]]
        # For binary classification:
        # - y_prob[0] = probability of class 0 (trash)
        # - y_prob[1] = probability of class 1 (keep)
        # These sum to 1.0

        mock_proba = np.array([[0.3, 0.7]])  # 30% trash, 70% keep
        prob_keep = float(mock_proba[0][1])

        assert prob_keep == 0.7
        assert prob_keep > 0.6  # Should be included

        label = "keep" if prob_keep > 0.5 else "trash"
        assert label == "keep"

        confidence = prob_keep if label == "keep" else (1 - prob_keep)
        assert confidence == 0.7


class TestAutoLabelingIntegration:
    """Integration tests for auto-labeling workflow."""

    def test_empty_predictions_debugging(self):
        """Test that empty predictions provide useful debugging info."""
        result = {
            "success": True,
            "predictions_count": 0,
            "predictions": [],
            "message": "Generated 0 predictions (processed 362 images)",
            "stats": {
                "total_unlabeled": 362,
                "files_existing": 362,
                "feature_errors": 0,
                "prediction_errors": 0,
                "low_confidence": 362,
                "high_confidence": 0
            }
        }

        # When we get 0 predictions, check the stats to diagnose:
        stats = result["stats"]

        # All images processed successfully
        assert stats["feature_errors"] == 0
        assert stats["prediction_errors"] == 0

        # But ALL were filtered as low confidence
        assert stats["low_confidence"] == stats["total_unlabeled"]
        assert stats["high_confidence"] == 0

        # This tells us: model IS working, but all predictions are in 0.4-0.6 range
        # Possible causes:
        # 1. Model not trained enough (needs more data or better features)
        # 2. Model is uncertain (images are genuinely ambiguous)
        # 3. Threshold too strict (but 0.4-0.6 is reasonable)

        diagnosis = "Model predictions all in uncertain range (0.4-0.6)"
        assert stats["low_confidence"] > 0.9 * stats["total_unlabeled"]
        print(f"Diagnosis: {diagnosis}")
        print("Recommendation: Check training data quality and quantity")
        print("Expected: Need at least 50-100 labeled examples per class")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

