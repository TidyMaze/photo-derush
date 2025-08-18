import os
import threading
import logging
from typing import Dict, Tuple
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ImageManager:
    """Manages thumbnail cache only (miniatures) to avoid memory growth from full-size images.
    Full images are opened on demand (not cached) and file handles closed immediately.
    Thread-safe for concurrent reads."""
    def __init__(self, max_thumbs: int = 1000):
        self._lock = threading.RLock()
        # thumb cache: (path, size) -> (mtime, Image)
        self._thumbs: "OrderedDict[Tuple[str, Tuple[int,int]], Tuple[float, Image.Image]]" = OrderedDict()
        self._max_thumbs = max_thumbs

    def _current_mtime(self, path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    # Ephemeral open of full-size image (not cached) --------------------------------
    def get_image(self, path: str):
        mtime = self._current_mtime(path)
        if mtime < 0:
            return None
        try:
            with Image.open(path) as img:
                img.load()  # force read then close
                ephem = img.copy()  # detach from underlying file
            return ephem
        except Exception as e:  # noqa: PERF203
            logger.warning("[ImageManager] Failed to open %s: %s", path, e)
            return None

    # Thumbnail handling -----------------------------------------------------------
    def _evict_if_needed(self):
        # Simple FIFO/LRU (OrderedDict preserves insertion; move_to_end on access)
        while len(self._thumbs) > self._max_thumbs:
            k, v = self._thumbs.popitem(last=False)
            try:
                # Help GC by closing image pointer if any
                if hasattr(v[1], 'fp') and getattr(v[1], 'fp', None):
                    try: v[1].fp.close()
                    except Exception: pass
            except Exception:
                pass
            logger.debug("[ImageManager] Evicted old thumbnail %s", k[0])

    def get_thumbnail(self, path: str, size: Tuple[int,int]):
        mtime = self._current_mtime(path)
        if mtime < 0:
            return None
        key = (path, size)
        with self._lock:
            cached = self._thumbs.get(key)
            if cached and cached[0] == mtime:
                # Promote to end (LRU)
                self._thumbs.move_to_end(key)
                return cached[1]
        # Need to (re)generate
        try:
            with Image.open(path) as src:
                src.load()
                # Work on a copy to avoid modifying src metadata; ensures closure
                work = src.copy()
        except Exception as e:  # noqa: PERF203
            logger.warning("[ImageManager] Failed to open for thumbnail %s: %s", path, e)
            return None
        try:
            work.thumbnail(size)
        except Exception as e:  # noqa: PERF203
            logger.warning("[ImageManager] Failed to build thumbnail %s: %s", path, e)
            return None
        # Persist thumbnail to disk (best effort)
        try:
            thumb_dir = os.path.join(os.path.dirname(path), 'thumbnails')
            os.makedirs(thumb_dir, exist_ok=True)
            thumb_path = os.path.join(thumb_dir, os.path.basename(path))
            if not (os.path.exists(thumb_path) and os.path.getmtime(thumb_path) >= mtime):
                work.save(thumb_path)
        except Exception as e:  # noqa: PERF203
            logger.debug("[ImageManager] Could not persist thumbnail for %s: %s", path, e)
        with self._lock:
            self._thumbs[key] = (mtime, work)
            self._evict_if_needed()
        return work

    def clear(self):
        with self._lock:
            self._thumbs.clear()
            logger.info("[ImageManager] Thumbnail cache cleared")

image_manager = ImageManager()
