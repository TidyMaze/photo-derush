"""Tests for pick_score calculation."""


from src.photo_grouping import compute_pick_score


def test_compute_pick_score_basic():
    """Pick score combines multiple quality metrics."""
    score = compute_pick_score(
        global_keep_score=0.8,
        sharpness=0.9,
        exposure_quality=0.7,
        noise_level=0.2,  # Low noise = good
        motion_blur=0.1,  # Low blur = good
    )

    # Should be positive and reasonable
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # With good metrics, score should be decent


def test_compute_pick_score_noise_inversion():
    """Noise level is inverted (lower noise = higher score)."""
    score_low_noise = compute_pick_score(
        global_keep_score=0.5,
        noise_level=0.1,  # Low noise
    )

    score_high_noise = compute_pick_score(
        global_keep_score=0.5,
        noise_level=0.9,  # High noise
    )

    assert score_low_noise > score_high_noise


def test_compute_pick_score_motion_blur_inversion():
    """Motion blur is inverted (lower blur = higher score)."""
    score_low_blur = compute_pick_score(
        global_keep_score=0.5,
        motion_blur=0.1,  # Low blur
    )

    score_high_blur = compute_pick_score(
        global_keep_score=0.5,
        motion_blur=0.9,  # High blur
    )

    assert score_low_blur > score_high_blur


def test_compute_pick_score_weights():
    """Custom weights can be provided."""
    score_default = compute_pick_score(
        global_keep_score=0.5,
        sharpness=1.0,
    )

    score_custom = compute_pick_score(
        global_keep_score=0.5,
        sharpness=1.0,
        weights={"global_keep": 0.1, "sharpness": 0.9, "exposure": 0.0, "noise": 0.0, "face": 0.0, "motion_blur": 0.0},
    )

    # Custom weights should change the score
    assert score_custom != score_default
    # With high sharpness weight, custom should be higher
    assert score_custom > score_default

