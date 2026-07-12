"""
Shared PIL Image cache - lightweight wrapper to avoid caching full images in memory.

This module provides convenience functions for opening images without caching
full decoded images in memory. The OS file system cache handles repeated opens efficiently.

Memory impact: Removed ~1.9GB of in-memory image cache (100 images × 19MB each).
Performance: OS file system cache provides similar benefits with minimal memory cost.
"""

import contextlib
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def get_cached_image(path: str):
    """
    Open an image file with a context manager to ensure it is closed and locks are released.
    """
    img = None
    try:
        img = Image.open(path)
        yield img
    except Exception as e:
        logger.warning(f"Failed to open image {path}: {e}")
        yield None
    finally:
        if img is not None:
            img.close()


@contextlib.contextmanager
def get_cached_image_for_exif(path: str):
    """
    Open an image file for EXIF extraction with a context manager to ensure locks are released.
    """
    img = None
    try:
        img = Image.open(path)
        yield img
    except Exception as e:
        logger.warning(f"Failed to open image for EXIF {path}: {e}")
        yield None
    finally:
        if img is not None:
            img.close()


def clear_image_cache():
    """
    Clear image cache (no-op, kept for API compatibility).
    """
    pass

