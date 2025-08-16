import os
import threading
import logging
from typing import Dict, Tuple
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class ImageManager:
    """Caches opened images and generated thumbnails (in-memory) with mtime validation.
    Thread-safe for concurrent reads.
    """
    def __init__(self):
        self._lock = threading.RLock()
        # original cache: path -> (mtime, Image)
        self._orig: Dict[str, Tuple[float, Image.Image]] = {}
        # thumb cache: (path, size) -> (mtime, Image)
        self._thumbs: Dict[Tuple[str, Tuple[int,int]], Tuple[float, Image.Image]] = {}

    def _current_mtime(self, path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return -1.0

    def get_image(self, path: str) -> Image.Image | None:
        mtime = self._current_mtime(path)
        if mtime < 0:
            return None
        with self._lock:
            cached = self._orig.get(path)
            if cached and cached[0] == mtime:
                return cached[1]
            try:
                img = Image.open(path)
                img.load()  # force load so we can close fp
                if hasattr(img, 'fp') and img.fp:
                    try:
                        img.fp.close()
                    except Exception:
                        pass
                    img.fp = None
                self._orig[path] = (mtime, img)
                return img
            except Exception as e:
                logger.warning("[ImageManager] Failed to open %s: %s", path, e)
                return None

    def get_thumbnail(self, path: str, size: Tuple[int,int]) -> Image.Image | None:
        mtime = self._current_mtime(path)
        if mtime < 0:
            return None
        key = (path, size)
        with self._lock:
            cached = self._thumbs.get(key)
            if cached and cached[0] == mtime:
                return cached[1]
        # Need to (re)create
        base = self.get_image(path)
        if base is None:
            return None
        try:
            # Work on a copy to preserve original object dimensions
            thumb = base.copy()
            thumb.thumbnail(size)
        except Exception as e:
            logger.warning("[ImageManager] Failed to make thumbnail for %s: %s", path, e)
            return None
        # Persist to disk thumbnail folder (mirrors previous behaviour) for reuse outside cache
        try:
            thumb_dir = os.path.join(os.path.dirname(path), 'thumbnails')
            os.makedirs(thumb_dir, exist_ok=True)
            thumb_path = os.path.join(thumb_dir, os.path.basename(path))
            if not (os.path.exists(thumb_path) and os.path.getmtime(thumb_path) >= mtime):
                thumb.save(thumb_path)
        except Exception as e:
            logger.info("[ImageManager] Could not persist thumbnail for %s: %s", path, e)
        with self._lock:
            self._thumbs[key] = (mtime, thumb)
        return thumb

    def clear(self):
        with self._lock:
            self._orig.clear()
            self._thumbs.clear()
            logger.info("[ImageManager] Caches cleared")

image_manager = ImageManager()
