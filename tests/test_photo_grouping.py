"""Tests for photo grouping (session, burst, near-duplicate detection)."""

import datetime
from dataclasses import dataclass

from src.photo_grouping import (
    detect_bursts,
    detect_sessions,
    group_near_duplicates,
    recommend_best_pick,
)


@dataclass
class PhotoInfo:
    """Test helper: photo metadata."""
    filename: str
    timestamp: datetime.datetime
    camera_id: str = "default"
    path: str = ""


def test_detect_sessions_time_gap():
    """Session detection: time gap > 30 min creates new session."""
    base_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
    photos = [
        PhotoInfo("img1.jpg", base_time, "camera1"),
        PhotoInfo("img2.jpg", base_time + datetime.timedelta(minutes=5), "camera1"),
        PhotoInfo("img3.jpg", base_time + datetime.timedelta(minutes=36), "camera1"),  # Gap > 30 min (35+1=36)
        PhotoInfo("img4.jpg", base_time + datetime.timedelta(minutes=40), "camera1"),
    ]

    sessions = detect_sessions(photos, session_gap_min=30)

    assert len(sessions) == 4  # One session ID per photo
    assert len(set(sessions)) == 2  # Two unique sessions
    assert sessions[0] == sessions[1]  # First two in same session
    assert sessions[1] != sessions[2]  # New session after gap
    assert sessions[2] == sessions[3]  # Last two in same session


def test_detect_sessions_camera_change():
    """Session detection: camera change creates new session."""
    base_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
    photos = [
        PhotoInfo("img1.jpg", base_time, "camera1"),
        PhotoInfo("img2.jpg", base_time + datetime.timedelta(minutes=5), "camera1"),
        PhotoInfo("img3.jpg", base_time + datetime.timedelta(minutes=10), "camera2"),  # Camera change
        PhotoInfo("img4.jpg", base_time + datetime.timedelta(minutes=15), "camera2"),
    ]

    sessions = detect_sessions(photos, session_gap_min=30)

    assert len(sessions) == 4
    assert sessions[0] == sessions[1]  # Same camera, same session
    assert sessions[1] != sessions[2]  # Camera change = new session
    assert sessions[2] == sessions[3]  # Same camera, same session


def test_detect_bursts():
    """Burst detection: time gap > 1 sec creates new burst."""
    base_time = datetime.datetime(2024, 1, 1, 12, 0, 0)
    photos = [
        PhotoInfo("img1.jpg", base_time, "camera1"),
        PhotoInfo("img2.jpg", base_time + datetime.timedelta(seconds=0.5), "camera1"),
        PhotoInfo("img3.jpg", base_time + datetime.timedelta(seconds=2.0), "camera1"),  # Gap > 1 sec
        PhotoInfo("img4.jpg", base_time + datetime.timedelta(seconds=2.3), "camera1"),
    ]

    bursts = detect_bursts(photos, burst_gap_sec=1.0)

    assert len(bursts) == 4
    assert bursts[0] == bursts[1]  # First two in same burst
    assert bursts[1] != bursts[2]  # New burst after gap
    assert bursts[2] == bursts[3]  # Last two in same burst


def test_group_near_duplicates_connected_components():
    """Near-duplicate grouping: similar hashes form connected components."""
    # Use exact hash matching for simplicity (imagehash format requires proper hex strings)
    # Photos 0,1,2 share same hash; 3,4 share same hash; 5 is unique
    photo_hashes = {
        "img0.jpg": "abc123",
        "img1.jpg": "abc123",  # Same as img0
        "img2.jpg": "abc123",  # Same as img0
        "img3.jpg": "def456",
        "img4.jpg": "def456",  # Same as img3
        "img5.jpg": "xyz999",  # Unique
    }

    def hash_fn(filename):
        return photo_hashes[filename]

    groups = group_near_duplicates(
        ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"],
        hash_fn,
        hamming_threshold=0,  # Exact match only (since we're using simple strings)
    )

    # Should have 3 groups: {0,1,2}, {3,4}, {5}
    unique_groups = set(groups)
    assert len(unique_groups) == 3
    assert groups[0] == groups[1] == groups[2]  # First 3 in same group
    assert groups[3] == groups[4]  # Next 2 in same group
    assert groups[5] not in [groups[0], groups[3]]  # Last is unique


def test_recommend_best_pick_exactly_one():
    """Best-pick recommendation: exactly one best per group."""
    # Group 0: photos 0,1,2
    # Group 1: photos 3,4
    # Group 2: photo 5 (singleton)
    groups = [0, 0, 0, 1, 1, 2]
    pick_scores = [0.5, 0.8, 0.6, 0.3, 0.9, 0.7]  # Best: idx 1 (0.8), idx 4 (0.9), idx 5 (0.7)

    best_flags = recommend_best_pick(groups, pick_scores)

    # Exactly one True per group
    assert best_flags[1] is True  # Best in group 0
    assert best_flags[4] is True  # Best in group 1
    assert best_flags[5] is True  # Best in group 2 (singleton)
    assert sum(best_flags) == 3  # Exactly 3 best picks (one per group)

    # Verify no other photos are marked best
    assert best_flags[0] is False
    assert best_flags[2] is False
    assert best_flags[3] is False


def test_recommend_best_pick_tie_breaker():
    """Best-pick recommendation: tie-breaker uses earlier timestamp."""
    groups = [0, 0]  # Two photos in same group
    pick_scores = [0.8, 0.8]  # Tie in score
    timestamps = [
        datetime.datetime(2024, 1, 1, 12, 0, 0),
        datetime.datetime(2024, 1, 1, 12, 0, 1),  # Later
    ]

    best_flags = recommend_best_pick(groups, pick_scores, timestamps=timestamps)

    # Earlier photo should win tie
    assert best_flags[0] is True
    assert best_flags[1] is False
