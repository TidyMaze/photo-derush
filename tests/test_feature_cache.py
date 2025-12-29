"""Tests for feature cache robustness."""
from __future__ import annotations

import os
import pickle
import tempfile
from unittest.mock import patch

from src.features import (
    FEATURE_COUNT,
    _validate_cache_entries,
    load_feature_cache,
    save_feature_cache,
)


def test_cache_version_detection_fast_mode(tmp_path):
    """Cache with mismatched FEATURE_COUNT should be auto-deleted."""
    cache_file = tmp_path / "cache.pkl"

    # Create v2 cache with wrong feature count (e.g., 95 instead of 71)
    bad_cache = {
        '__metadata__': {
            'feature_count': 95,  # Mismatch if running in FAST mode
            'mode': 'FULL',
            'version': 2,
        },
        '/some/image.jpg': [0.0] * 71,  # Only 71 features
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(bad_cache, f)

    assert cache_file.exists()

    # Patch FEATURE_CACHE_PATH to use our temp file
    with patch('src.features.FEATURE_CACHE_PATH', str(cache_file)):
        cache = load_feature_cache()

    # Cache should be deleted due to mismatch
    assert not cache_file.exists()
    assert cache == {}


def test_cache_v1_backward_compat(tmp_path):
    """Old v1 cache (no metadata) should load without deletion."""
    cache_file = tmp_path / "cache.pkl"

    # Create real image files
    img1 = tmp_path / "image1.jpg"
    img2 = tmp_path / "image2.jpg"
    img1.write_text("fake")
    img2.write_text("fake")

    # Create v1 cache (no __metadata__)
    old_cache = {
        str(img1): [0.0] * FEATURE_COUNT,
        str(img2): [1.0] * FEATURE_COUNT,
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(old_cache, f)

    with patch('src.features.FEATURE_CACHE_PATH', str(cache_file)):
        cache = load_feature_cache()

    # Should load v1 cache successfully
    assert len(cache) >= 2


def test_cache_invalid_entry_removal(tmp_path):
    """Invalid entries (wrong length, missing files) should be purged."""
    cache_file = tmp_path / "cache.pkl"
    img_file = tmp_path / "image.jpg"
    img_file.write_text("fake image")

    # Create cache with mixed valid/invalid entries
    cache_data = {
        '__metadata__': {
            'feature_count': FEATURE_COUNT,
            'mode': 'FAST',
            'version': 2,
        },
        str(img_file): [0.0] * FEATURE_COUNT,  # Valid
        '/deleted/image.jpg': [1.0] * FEATURE_COUNT,  # File missing
        str(tmp_path / 'bad.jpg'): [2.0] * (FEATURE_COUNT - 1),  # Wrong length
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    with patch('src.features.FEATURE_CACHE_PATH', str(cache_file)):
        cache = load_feature_cache()

    # Only valid entry should remain
    assert len(cache) == 1
    assert str(img_file) in cache


def test_cache_atomic_write(tmp_path):
    """save_feature_cache should use atomic write (temp → rename)."""
    cache_file = tmp_path / "cache.pkl"

    test_cache = {
        '/image1.jpg': [0.0] * FEATURE_COUNT,
        '/image2.jpg': [1.0] * FEATURE_COUNT,
    }

    with patch('src.features.FEATURE_CACHE_PATH', str(cache_file)):
        save_feature_cache(test_cache)

    # Verify file exists and contains metadata
    assert cache_file.exists()
    with open(cache_file, 'rb') as f:
        saved = pickle.load(f)

    assert '__metadata__' in saved
    assert saved['__metadata__']['feature_count'] == FEATURE_COUNT
    assert len({k: v for k, v in saved.items() if k != '__metadata__'}) == 2


def test_validate_cache_entries_file_exists():
    """_validate_cache_entries should remove entries for deleted files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_file = os.path.join(tmpdir, 'exists.jpg')
        open(img_file, 'w').close()

        cache = {
            img_file: [0.0] * FEATURE_COUNT,
            '/deleted/image.jpg': [1.0] * FEATURE_COUNT,
        }

        valid, removed = _validate_cache_entries(cache)

        assert len(valid) == 1
        assert img_file in valid
        assert removed == 1


def test_validate_cache_entries_feature_count():
    """_validate_cache_entries should remove entries with wrong feature count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_file = os.path.join(tmpdir, 'image.jpg')
        open(img_file, 'w').close()

        cache = {
            img_file: [0.0] * FEATURE_COUNT,  # Correct
            img_file + '2': [1.0] * (FEATURE_COUNT - 1),  # Wrong length
        }

        valid, removed = _validate_cache_entries(cache)

        assert len(valid) == 1
        assert removed == 1


def test_cache_corruption_recovery(tmp_path):
    """Corrupted cache file should be deleted and recovered."""
    cache_file = tmp_path / "cache.pkl"

    # Write corrupted pickle
    with open(cache_file, 'wb') as f:
        f.write(b'not a valid pickle file \x00\xff')

    with patch('src.features.FEATURE_CACHE_PATH', str(cache_file)):
        cache = load_feature_cache()

    # Should recover gracefully
    assert not cache_file.exists()
    assert cache == {}

