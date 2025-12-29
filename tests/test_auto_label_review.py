"""Tests for human-in-the-loop auto-label review system."""
from __future__ import annotations

import pytest

from src.auto_label_review import AutoLabelReviewManager


@pytest.fixture
def review_manager(tmp_path):
    """Create a temporary review manager."""
    manager = AutoLabelReviewManager(str(tmp_path), confidence_threshold=0.9)
    return manager


def test_flag_for_review_above_threshold(review_manager):
    """High-confidence predictions should be flagged."""
    flagged = review_manager.flag_for_review('image.jpg', 'keep', 0.95)
    assert flagged is True

    candidates = review_manager.get_review_candidates()
    assert len(candidates) == 1
    assert candidates[0]['filename'] == 'image.jpg'
    assert candidates[0]['predicted_label'] == 'keep'
    assert candidates[0]['confidence'] == 0.95


def test_flag_for_review_below_threshold(review_manager):
    """Low-confidence predictions should not be flagged."""
    flagged = review_manager.flag_for_review('image.jpg', 'keep', 0.7)
    assert flagged is False

    candidates = review_manager.get_review_candidates()
    assert len(candidates) == 0


def test_confirm_label(review_manager):
    """User should be able to confirm auto-labels."""
    review_manager.flag_for_review('image.jpg', 'keep', 0.95)

    # Confirm the label
    confirmed = review_manager.confirm_label('image.jpg', 'keep')
    assert confirmed is True

    # Should no longer appear in pending candidates
    candidates = review_manager.get_review_candidates()
    assert len(candidates) == 0

    # Should appear in confirmed labels
    confirmed_labels = review_manager.get_confirmed_labels()
    assert confirmed_labels['image.jpg'] == 'keep'


def test_reject_label(review_manager):
    """User should be able to reject auto-labels."""
    review_manager.flag_for_review('image.jpg', 'keep', 0.95)

    # Reject by passing empty string
    confirmed = review_manager.confirm_label('image.jpg', '')
    assert confirmed is True

    # Should not appear in confirmed labels
    confirmed_labels = review_manager.get_confirmed_labels()
    assert 'image.jpg' not in confirmed_labels


def test_correct_prediction(review_manager):
    """User should be able to correct predictions."""
    review_manager.flag_for_review('image.jpg', 'keep', 0.92)

    # User corrects to trash
    review_manager.confirm_label('image.jpg', 'trash')

    confirmed_labels = review_manager.get_confirmed_labels()
    assert confirmed_labels['image.jpg'] == 'trash'


def test_persistence(tmp_path):
    """Review data should persist to disk."""
    manager1 = AutoLabelReviewManager(str(tmp_path), confidence_threshold=0.9)
    manager1.flag_for_review('image1.jpg', 'keep', 0.95)
    manager1.flag_for_review('image2.jpg', 'trash', 0.91)
    manager1.confirm_label('image1.jpg', 'keep')

    # Load from same directory
    manager2 = AutoLabelReviewManager(str(tmp_path), confidence_threshold=0.9)

    # Should have persisted
    candidates = manager2.get_review_candidates()
    assert len(candidates) == 1
    assert candidates[0]['filename'] == 'image2.jpg'

    confirmed = manager2.get_confirmed_labels()
    assert confirmed['image1.jpg'] == 'keep'


def test_review_stats(review_manager):
    """Stats should accurately reflect review progress."""
    review_manager.flag_for_review('img1.jpg', 'keep', 0.95)
    review_manager.flag_for_review('img2.jpg', 'keep', 0.92)
    review_manager.flag_for_review('img3.jpg', 'trash', 0.91)

    review_manager.confirm_label('img1.jpg', 'keep')
    review_manager.confirm_label('img2.jpg', '')  # reject

    stats = review_manager.get_review_stats()
    assert stats['total_flagged'] == 3
    assert stats['pending'] == 1
    assert stats['reviewed'] == 2
    assert stats['confirmed'] == 1
    assert stats['rejected'] == 1


def test_clear_reviewed(review_manager):
    """Clearing reviewed items should remove them from tracking."""
    review_manager.flag_for_review('img1.jpg', 'keep', 0.95)
    review_manager.flag_for_review('img2.jpg', 'trash', 0.91)

    review_manager.confirm_label('img1.jpg', 'keep')

    stats_before = review_manager.get_review_stats()
    assert stats_before['reviewed'] == 1

    review_manager.clear_reviewed()

    stats_after = review_manager.get_review_stats()
    assert stats_after['total_flagged'] == 1  # Only pending item
    assert stats_after['pending'] == 1


def test_add_confirmed_to_repository(review_manager):
    """Confirmed labels should be added to repository."""
    from unittest.mock import MagicMock

    repo = MagicMock()

    review_manager.flag_for_review('keep_img.jpg', 'keep', 0.95)
    review_manager.flag_for_review('trash_img.jpg', 'trash', 0.93)
    review_manager.flag_for_review('reject_img.jpg', 'keep', 0.91)

    review_manager.confirm_label('keep_img.jpg', 'keep')
    review_manager.confirm_label('trash_img.jpg', 'trash')
    review_manager.confirm_label('reject_img.jpg', '')  # rejected

    added = review_manager.add_confirmed_to_repository(repo)

    assert added == 2
    assert repo.set_state.call_count == 2

    # Verify correct calls
    calls = repo.set_state.call_args_list
    assert ('keep_img.jpg', 'keep') in [call[0] for call in calls]
    assert ('trash_img.jpg', 'trash') in [call[0] for call in calls]


def test_multiple_predictions_same_image(review_manager):
    """Subsequent predictions should overwrite previous ones."""
    review_manager.flag_for_review('image.jpg', 'keep', 0.95)
    assert review_manager.get_review_candidates()[0]['predicted_label'] == 'keep'

    # New prediction comes in
    review_manager.flag_for_review('image.jpg', 'trash', 0.92)
    assert review_manager.get_review_candidates()[0]['predicted_label'] == 'trash'
    assert review_manager.get_review_candidates()[0]['confidence'] == 0.92


def test_confirm_nonexistent_image(review_manager):
    """Confirming non-existent image should return False."""
    confirmed = review_manager.confirm_label('nonexistent.jpg', 'keep')
    assert confirmed is False

