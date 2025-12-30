"""Tests for pick_score calculation."""


from src.photo_grouping import compute_pick_score


def test_compute_pick_score_basic():
    """Pick score equals keep score directly (no heuristic)."""
    score = compute_pick_score(
        global_keep_score=0.8,
        sharpness=0.9,
        exposure_quality=0.7,
        noise_level=0.2,  # Low noise = good
        motion_blur=0.1,  # Low blur = good
    )

    # Should equal the keep score (other params ignored)
    assert score == 0.8
    assert 0.0 <= score <= 1.0


def test_compute_pick_score_ignores_other_params():
    """Other parameters are ignored - only keep score is used."""
    score_low_noise = compute_pick_score(
        global_keep_score=0.5,
        noise_level=0.1,  # Low noise
    )

    score_high_noise = compute_pick_score(
        global_keep_score=0.5,
        noise_level=0.9,  # High noise
    )

    # Same keep score = same pick score (noise is ignored)
    assert score_low_noise == score_high_noise == 0.5


def test_compute_pick_score_direct_mapping():
    """Pick score equals keep score directly."""
    score_low_blur = compute_pick_score(
        global_keep_score=0.5,
        motion_blur=0.1,  # Low blur
    )

    score_high_blur = compute_pick_score(
        global_keep_score=0.5,
        motion_blur=0.9,  # High blur
    )

    # Same keep score = same pick score (motion blur is ignored)
    assert score_low_blur == score_high_blur == 0.5


def test_compute_pick_score_weights_ignored():
    """Custom weights are ignored - only keep score matters."""
    score_default = compute_pick_score(
        global_keep_score=0.5,
        sharpness=1.0,
    )

    score_custom = compute_pick_score(
        global_keep_score=0.5,
        sharpness=1.0,
        weights={"global_keep": 0.1, "sharpness": 0.9, "exposure": 0.0, "noise": 0.0, "face": 0.0, "motion_blur": 0.0},
    )

    # Weights are ignored, same keep score = same pick score
    assert score_custom == score_default == 0.5

