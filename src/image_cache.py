"""
Shared PIL Image cache - lightweight wrapper to avoid caching full images in memory.

This module provides convenience functions for opening images without caching
full decoded images in memory. The OS file system cache handles repeated opens efficiently.

Memory impact: Removed ~1.9GB of in-memory image cache (100 images × 19MB each).
Performance: OS file system cache provides similar benefits with minimal memory cost.
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def get_cached_image(path: str) -> Optional[Image.Image]:
    """
    Open an image file (convenience function, no in-memory caching).
    
    This function opens images without caching full decoded images in memory.
    The OS file system cache handles repeated opens efficiently.
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL Image object (lazy-loaded, not decoded until needed), or None if opening failed
    """
    try:
        # Open image lazily - OS file system cache handles repeated opens
        # PIL Image.open() is lazy by default - doesn't decode until load() is called
        return Image.open(path)
    except Exception as e:
        logger.warning(f"Failed to open image {path}: {e}")
        return None


def get_cached_image_for_exif(path: str) -> Optional[Image.Image]:
    """
    Open an image file for EXIF extraction (convenience function, no in-memory caching).
    
    This function opens images without caching full decoded images in memory.
    EXIF extraction doesn't require full image decode, so this is very memory-efficient.
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL Image object (lazy-loaded, not decoded), or None if opening failed
    """
    try:
        # Open image lazily - EXIF extraction doesn't need full decode
        # OS file system cache handles repeated opens efficiently
        return Image.open(path)
    except Exception as e:
        logger.warning(f"Failed to open image for EXIF {path}: {e}")
        return None


def clear_image_cache():
    """
    Clear image cache (no-op, kept for API compatibility).
    
    Since we no longer cache images in memory, this is a no-op.
    """
    pass

