"""
Tests for PIL Image cache functions.

Tests verify that get_cached_image() and get_cached_image_for_exif() work correctly.
Note: These functions no longer cache images in memory to reduce memory usage.
OS file system cache handles repeated opens efficiently.
"""

import os
import tempfile

import pytest
from PIL import Image

from src.image_cache import clear_image_cache, get_cached_image, get_cached_image_for_exif


@pytest.fixture
def test_image_path():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(f.name, "PNG")
        yield f.name
    os.unlink(f.name)


def test_get_cached_image(test_image_path):
    """Test that get_cached_image opens images correctly."""
    # First call - should open image
    img1 = get_cached_image(test_image_path)
    assert img1 is not None
    assert isinstance(img1, Image.Image)
    assert img1.size == (100, 100)
    
    # Second call - should open again (no in-memory cache, but OS cache helps)
    img2 = get_cached_image(test_image_path)
    assert img2 is not None
    assert isinstance(img2, Image.Image)
    assert img2.size == (100, 100)
    
    # Should be different objects (new opens each time)
    assert img1 is not img2


def test_get_cached_image_for_exif(test_image_path):
    """Test that get_cached_image_for_exif works for EXIF extraction."""
    img = get_cached_image_for_exif(test_image_path)
    assert img is not None
    assert isinstance(img, Image.Image)
    
    # Should be able to extract EXIF (even if empty for PNG)
    # Use getexif() for newer PIL versions, fallback to _getexif() for older
    if hasattr(img, "getexif"):
        exif = img.getexif()
    elif hasattr(img, "_getexif"):
        exif = img._getexif()  # type: ignore[attr-defined]
    else:
        exif = None
    # EXIF might be None for PNG, that's OK


def test_get_cached_image_invalid_path():
    """Test that get_cached_image handles invalid paths gracefully."""
    invalid_path = "/nonexistent/path/image.png"
    img = get_cached_image(invalid_path)
    assert img is None


def test_get_cached_image_for_exif_invalid_path():
    """Test that get_cached_image_for_exif handles invalid paths gracefully."""
    invalid_path = "/nonexistent/path/image.png"
    img = get_cached_image_for_exif(invalid_path)
    assert img is None


def test_clear_image_cache():
    """Test that clear_image_cache is a no-op (API compatibility)."""
    # Should not raise
    clear_image_cache()
    clear_image_cache()


def test_get_cached_image_lazy_loading(test_image_path):
    """Test that images are opened lazily (not decoded until needed)."""
    img = get_cached_image(test_image_path)
    assert img is not None
    
    # Image should be opened but not necessarily decoded
    # Accessing size should work (metadata only, no decode)
    assert img.size == (100, 100)
    
    # Actually loading the image should work
    img.load()
    assert img.size == (100, 100)
