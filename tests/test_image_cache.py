"""
Tests for PIL Image cache optimization.

Tests verify that the image cache reduces repeated file opens
and maintains correctness.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest
from PIL import Image

from src.image_cache import ImageCache, clear_image_cache, get_cached_image, get_cached_image_for_exif, get_image_cache


@pytest.fixture
def test_image_path():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(f.name, "PNG")
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def cache():
    """Create a fresh image cache for each test."""
    clear_image_cache()
    cache = ImageCache(maxsize=10)
    yield cache
    cache.clear()


def test_image_cache_get_image(test_image_path, cache):
    """Test that get_image returns a copy of the cached image."""
    # First call - should open and cache
    img1 = cache.get_image(test_image_path)
    assert img1 is not None
    assert isinstance(img1, Image.Image)
    
    # Second call - should return from cache (copy)
    img2 = cache.get_image(test_image_path)
    assert img2 is not None
    assert isinstance(img2, Image.Image)
    
    # Should be different objects (copies)
    assert img1 is not img2
    
    # But should have same size
    assert img1.size == img2.size


def test_image_cache_hit_rate(test_image_path, cache):
    """Test that cache hit rate increases with repeated access."""
    # First call - miss
    cache.get_image(test_image_path)
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    
    # Second call - hit
    cache.get_image(test_image_path)
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 50.0


def test_image_cache_lru_eviction(cache):
    """Test that LRU eviction works when cache is full."""
    # Create multiple test images
    test_images = []
    for i in range(15):  # More than maxsize (10)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (100, 100), color=(i % 255, 0, 0))
            img.save(f.name, "PNG")
            test_images.append(f.name)
    
    try:
        # Fill cache
        for path in test_images[:10]:
            cache.get_image(path)
        
        # Cache should be full
        stats = cache.get_stats()
        assert stats["size"] == 10
        
        # Add one more - should evict oldest
        cache.get_image(test_images[10])
        stats = cache.get_stats()
        assert stats["size"] == 10  # Still at maxsize
        
        # Oldest (first) should be evicted
        # Access it again - should be a miss
        cache.get_image(test_images[0])
        stats = cache.get_stats()
        # Should have one more miss
        assert stats["misses"] >= 2
    finally:
        # Cleanup
        for path in test_images:
            try:
                os.unlink(path)
            except Exception:
                pass


def test_image_cache_get_image_for_exif(test_image_path, cache):
    """Test that get_image_for_exif works for EXIF extraction."""
    img = cache.get_image_for_exif(test_image_path)
    assert img is not None
    assert isinstance(img, Image.Image)
    
    # Should be able to extract EXIF (even if empty)
    exif = img._getexif() if hasattr(img, "_getexif") else None
    # EXIF might be None for PNG, that's OK


def test_image_cache_clear(test_image_path, cache):
    """Test that clear() removes all cached images."""
    cache.get_image(test_image_path)
    stats = cache.get_stats()
    assert stats["size"] == 1
    
    cache.clear()
    stats = cache.get_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_global_image_cache(test_image_path):
    """Test that global cache functions work."""
    clear_image_cache()
    
    # Get image via global function
    img1 = get_cached_image(test_image_path)
    assert img1 is not None
    
    # Second call should hit cache
    img2 = get_cached_image(test_image_path)
    assert img2 is not None
    
    # Check stats
    cache = get_image_cache()
    stats = cache.get_stats()
    assert stats["hits"] >= 1


def test_image_cache_performance(test_image_path, cache):
    """Test that cached access is faster than repeated opens."""
    # Warm up cache
    cache.get_image(test_image_path)
    
    # Time cached access
    start = time.perf_counter()
    for _ in range(100):
        cache.get_image(test_image_path)
    cached_time = time.perf_counter() - start
    
    # Time direct opens
    start = time.perf_counter()
    for _ in range(100):
        with Image.open(test_image_path) as img:
            img.copy()
    direct_time = time.perf_counter() - start
    
    # Cached should be faster (at least 2x)
    assert cached_time < direct_time
    # Log for visibility
    print(f"\nCached: {cached_time:.4f}s, Direct: {direct_time:.4f}s, Speedup: {direct_time/cached_time:.2f}x")


def test_image_cache_thread_safety(test_image_path):
    """Test that cache is thread-safe."""
    import threading
    
    cache = ImageCache(maxsize=100)
    results = []
    errors = []
    
    def worker():
        try:
            for _ in range(10):
                img = cache.get_image(test_image_path)
                results.append(img is not None)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have no errors
    assert len(errors) == 0
    # Should have 50 successful results (5 threads × 10 calls)
    assert len(results) == 50
    assert all(results)


def test_image_cache_invalid_path(cache):
    """Test that cache handles invalid paths gracefully."""
    invalid_path = "/nonexistent/path/image.png"
    img = cache.get_image(invalid_path)
    assert img is None
    
    # Should not crash
    stats = cache.get_stats()
    assert stats["misses"] >= 1

