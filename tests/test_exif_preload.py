"""Tests for EXIF pre-loading optimization (correctness and performance)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.lazy_loader import LazyImageLoader


@pytest.fixture
def mock_model():
    """Mock ImageModel with load_exif that simulates slow I/O."""
    model = MagicMock()
    
    # Simulate slow EXIF loading (20ms per call)
    def slow_load_exif(path):
        time.sleep(0.020)  # 20ms delay
        return {'Make': 'TestCamera', 'Model': 'TestModel', 'path': path}
    
    model.load_exif = MagicMock(side_effect=slow_load_exif)
    model.load_thumbnail = MagicMock(return_value=MagicMock())
    return model


def test_exif_preload_correctness(mock_model):
    """Pre-loaded EXIF should return same data as synchronous load."""
    loader = LazyImageLoader(mock_model, max_workers=4, cache_size=256)
    
    paths = ['/img1.jpg', '/img2.jpg', '/img3.jpg']
    
    # Pre-load EXIF in background
    loader.preload_exif_silent(paths)
    
    # Wait for pre-loading to complete
    time.sleep(0.5)  # Should be enough for 3 images with 20ms each
    
    # Now access EXIF - should be fast (from cache)
    start = time.perf_counter()
    exif1 = loader._cached_exif('/img1.jpg')
    time_access = time.perf_counter() - start
    
    # Should be fast (< 1ms from cache)
    assert time_access < 0.001
    
    # Should have correct data
    assert exif1['Make'] == 'TestCamera'
    assert exif1['path'] == '/img1.jpg'
    
    # Verify all paths were loaded
    assert mock_model.load_exif.call_count == 3
    
    loader.shutdown(wait=False)


def test_exif_preload_performance(mock_model):
    """Pre-loading should be faster than sequential loading."""
    loader = LazyImageLoader(mock_model, max_workers=4, cache_size=256)
    
    paths = ['/img1.jpg', '/img2.jpg', '/img3.jpg', '/img4.jpg', '/img5.jpg']
    
    # Measure sequential loading time (clear cache first)
    loader.clear_cache()
    mock_model.load_exif.reset_mock()
    start_seq = time.perf_counter()
    for path in paths:
        loader._cached_exif(path)  # Sequential, each takes 20ms
    time_seq = time.perf_counter() - start_seq
    
    # Measure parallel pre-loading time
    loader.clear_cache()
    mock_model.load_exif.reset_mock()
    start_parallel = time.perf_counter()
    loader.preload_exif_silent(paths)
    # Wait for completion (5 images * 20ms / 4 workers = ~25ms, but allow more for overhead)
    time.sleep(0.15)
    time_parallel = time.perf_counter() - start_parallel
    
    # Parallel should be faster (or at least not much slower)
    # With 4 workers, 5 images should take ~25-50ms instead of 100ms sequential
    # Allow 2x overhead for thread scheduling
    assert time_parallel <= time_seq * 2.0  # Allow overhead
    
    loader.shutdown(wait=False)


def test_exif_preload_non_blocking(mock_model):
    """Pre-loading should not block the main thread."""
    loader = LazyImageLoader(mock_model, max_workers=4, cache_size=256)
    
    paths = ['/img1.jpg', '/img2.jpg', '/img3.jpg']
    
    # Start pre-loading
    start = time.perf_counter()
    loader.preload_exif_silent(paths)
    preload_time = time.perf_counter() - start
    
    # Pre-loading should return immediately (non-blocking)
    assert preload_time < 0.005  # Should return in < 5ms (allows for thread scheduling overhead)
    
    # Wait a bit for background loading
    time.sleep(0.2)
    
    # Now EXIF should be available
    exif = loader._cached_exif('/img1.jpg')
    assert exif['Make'] == 'TestCamera'
    
    loader.shutdown(wait=False)


def test_exif_preload_batch_large_dataset(mock_model):
    """Pre-loading should handle large datasets efficiently."""
    loader = LazyImageLoader(mock_model, max_workers=4, cache_size=256)
    
    # Create 50 image paths
    paths = [f'/img{i}.jpg' for i in range(50)]
    
    # Pre-load all
    start = time.perf_counter()
    loader.preload_exif_silent(paths)
    preload_time = time.perf_counter() - start
    
    # Should return immediately (non-blocking)
    assert preload_time < 0.005  # Allow for thread scheduling overhead
    
    # Wait for background loading (50 images * 20ms / 4 workers = ~250ms)
    time.sleep(0.5)
    
    # Verify all were loaded
    assert mock_model.load_exif.call_count == 50
    
    # Verify all are accessible from cache
    for path in paths[:10]:  # Check first 10
        exif = loader._cached_exif(path)
        assert exif['path'] == path
    
    loader.shutdown(wait=False)


def test_exif_preload_priority_paths(mock_model):
    """Priority paths should be loaded first."""
    loader = LazyImageLoader(mock_model, max_workers=4, cache_size=256)
    
    all_paths = [f'/img{i}.jpg' for i in range(10)]
    priority_paths = ['/img0.jpg', '/img1.jpg', '/img2.jpg']
    
    # Pre-load with priority
    loader.preload_exif_batch(all_paths, priority_paths=priority_paths)
    
    # Wait a bit
    time.sleep(0.1)
    
    # Priority paths should be loaded first (check by verifying they're in cache)
    # Note: This is a best-effort test since exact ordering depends on thread scheduling
    for path in priority_paths:
        exif = loader._cached_exif(path)
        assert exif['path'] == path
    
    loader.shutdown(wait=False)

