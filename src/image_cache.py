"""
Shared PIL Image cache to reduce repeated file opens.

This module provides a thread-safe LRU cache for PIL Image objects,
reducing the overhead of opening the same image file multiple times
for different purposes (EXIF extraction, thumbnails, features, etc.).

Expected impact: 39.2s → ~10-15s (60-75% reduction in PIL.Image.open time)
"""

import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ImageCache:
    """
    Thread-safe LRU cache for PIL Image objects.
    
    Images are opened once and cached, with copies provided to callers
    to avoid modification issues. The cache uses LRU eviction to manage memory.
    """
    
    def __init__(self, maxsize: int = 100):
        """
        Initialize the image cache.
        
        Args:
            maxsize: Maximum number of images to cache (default: 100)
        """
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._lock = threading.RLock()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
    
    def get_image(self, path: str) -> Optional[Image.Image]:
        """
        Get a copy of the image from cache, or open and cache it if not present.
        
        Returns a copy of the cached image so modifications don't affect the cache.
        If the image is not in cache, it's opened, cached, and a copy is returned.
        
        Args:
            path: Path to the image file
            
        Returns:
            PIL Image object (copy), or None if opening failed
        """
        with self._lock:
            # Check cache first
            if path in self._cache:
                self._hits += 1
                cached_img = self._cache[path]
                # Move to end (most recently used)
                self._cache.move_to_end(path)
                # Return a copy to avoid modification issues
                try:
                    return cached_img.copy()
                except Exception as e:
                    logger.warning(f"Failed to copy cached image {path}: {e}")
                    # Remove corrupted entry
                    del self._cache[path]
                    # Fall through to open fresh
            
            # Cache miss - open and cache
            self._misses += 1
            try:
                img = Image.open(path)
                # Store in cache
                self._cache[path] = img
                
                # Evict oldest if cache is full
                if len(self._cache) > self._maxsize:
                    oldest_path, oldest_img = self._cache.popitem(last=False)
                    try:
                        oldest_img.close()
                    except Exception:
                        pass
                
                # Return a copy
                return img.copy()
            except Exception as e:
                logger.warning(f"Failed to open image {path}: {e}")
                return None
    
    def get_image_for_exif(self, path: str) -> Optional[Image.Image]:
        """
        Get image for EXIF extraction (doesn't need full decode).
        
        This is optimized for EXIF extraction which doesn't require
        the image to be fully decoded.
        
        Args:
            path: Path to the image file
            
        Returns:
            PIL Image object (copy), or None if opening failed
        """
        # For EXIF, we can use lazy loading
        with self._lock:
            if path in self._cache:
                self._hits += 1
                cached_img = self._cache[path]
                self._cache.move_to_end(path)
                # For EXIF, we don't need a copy - just return the cached one
                # But we need to ensure it's not modified, so return a copy anyway
                return cached_img.copy()
            
            # Cache miss
            self._misses += 1
            try:
                img = Image.open(path)
                # Don't load() - keep it lazy for EXIF extraction
                self._cache[path] = img
                
                # Evict if needed
                if len(self._cache) > self._maxsize:
                    oldest_path, oldest_img = self._cache.popitem(last=False)
                    try:
                        oldest_img.close()
                    except Exception:
                        pass
                
                return img.copy()
            except Exception as e:
                logger.warning(f"Failed to open image for EXIF {path}: {e}")
                return None
    
    def clear(self):
        """Clear the cache and close all cached images."""
        with self._lock:
            for img in self._cache.values():
                try:
                    img.close()
                except Exception:
                    pass
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# Global image cache instance
_global_image_cache: Optional[ImageCache] = None
_cache_lock = threading.Lock()


def get_image_cache() -> ImageCache:
    """Get or create the global image cache instance."""
    global _global_image_cache
    if _global_image_cache is None:
        with _cache_lock:
            if _global_image_cache is None:
                _global_image_cache = ImageCache(maxsize=100)
    return _global_image_cache


def clear_image_cache():
    """Clear the global image cache."""
    global _global_image_cache
    if _global_image_cache is not None:
        _global_image_cache.clear()


def get_cached_image(path: str) -> Optional[Image.Image]:
    """
    Get a cached image (convenience function).
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL Image object (copy), or None if opening failed
    """
    return get_image_cache().get_image(path)


def get_cached_image_for_exif(path: str) -> Optional[Image.Image]:
    """
    Get a cached image for EXIF extraction (convenience function).
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL Image object (copy), or None if opening failed
    """
    return get_image_cache().get_image_for_exif(path)

