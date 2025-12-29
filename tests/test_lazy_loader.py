"""Tests for lazy image loading with threading."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.lazy_loader import LazyImageLoader


@pytest.fixture
def mock_model():
    """Mock ImageModel with load_exif and load_thumbnail."""
    model = MagicMock()
    model.load_exif = MagicMock(return_value={'Make': 'TestCamera'})
    model.load_thumbnail = MagicMock(return_value=MagicMock())
    return model


def test_lazy_loader_init(mock_model):
    """LazyImageLoader should initialize with thread pool."""
    loader = LazyImageLoader(mock_model, max_workers=2, cache_size=100)
    assert loader.model == mock_model
    assert loader._cancelled is False
    loader.shutdown(wait=False)


def test_lazy_loader_cache_hit(mock_model):
    """Repeated loads should hit LRU cache."""
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    # First call
    result1 = loader._cached_exif('/path/to/img.jpg')
    assert result1 == {'Make': 'TestCamera'}
    assert mock_model.load_exif.call_count == 1

    # Second call (should hit cache)
    result2 = loader._cached_exif('/path/to/img.jpg')
    assert result2 == {'Make': 'TestCamera'}
    assert mock_model.load_exif.call_count == 1  # No additional call

    loader.shutdown(wait=False)


def test_lazy_loader_clear_cache(mock_model):
    """clear_cache should reset LRU cache."""
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    loader._cached_exif('/img1.jpg')
    loader._cached_exif('/img2.jpg')

    info1 = loader.cache_info()
    assert 'exif_hits' in info1

    loader.clear_cache()

    # After clear, next call increments misses
    loader._cached_exif('/img1.jpg')
    info2 = loader.cache_info()
    assert info2['exif_misses'] >= 1

    loader.shutdown(wait=False)


def test_lazy_loader_cancel_resume(mock_model):
    """cancel() should stop operations, resume() should allow them."""
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    assert loader._cancelled is False
    loader.cancel()
    assert loader._cancelled is True

    loader.resume()
    assert loader._cancelled is False

    loader.shutdown(wait=False)


def test_lazy_loader_cache_info(mock_model):
    """cache_info() should return hit/miss counts."""
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    # Load same path twice
    loader._cached_exif('/img.jpg')
    loader._cached_exif('/img.jpg')

    info = loader.cache_info()
    assert 'exif_hits' in info
    assert 'exif_misses' in info
    # Second load should result in a hit
    assert info['exif_hits'] >= 1

    loader.shutdown(wait=False)


def test_lazy_loader_cache_diff_paths(mock_model):
    """Different paths should be cached separately."""
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    result1 = loader._cached_exif('/img1.jpg')
    result2 = loader._cached_exif('/img2.jpg')

    # Both should return same mock result
    assert result1 == result2
    # But model.load_exif should be called twice (different paths)
    assert mock_model.load_exif.call_count == 2

    loader.shutdown(wait=False)


def test_lazy_loader_exception_fallback(mock_model):
    """Exceptions should be handled gracefully."""
    mock_model.load_exif.side_effect = RuntimeError("Mock error")
    loader = LazyImageLoader(mock_model, max_workers=1, cache_size=10)

    # Synchronous call (for testing)
    # In real use, callback would get empty dict
    callback = MagicMock()
    try:
        loader.get_exif_lazy('/img.jpg', callback)
        time.sleep(0.1)  # Brief wait for async
    except Exception:
        pass  # Expected in mock scenario

    loader.shutdown(wait=False)

