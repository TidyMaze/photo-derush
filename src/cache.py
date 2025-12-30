import hashlib
import logging
import os

from PIL import Image, PngImagePlugin

from .cache_config import is_cache_disabled

CACHE_DIR = os.path.expanduser("~/.photo-derush-cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# PERFORMANCE: Cache stat results to reduce I/O (507 calls -> significant savings)
_stat_cache = {}
_stat_cache_time = {}

def _thumbnail_cache_key(path):
    # Cache stat results for 1 second to avoid repeated I/O
    import time
    current_time = time.time()
    
    if is_cache_disabled():
        # When cache disabled, always compute fresh stat
        try:
            stat = os.stat(path)
            key = f"{path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.sha256(key.encode()).hexdigest()
        except Exception:
            return hashlib.sha256(path.encode()).hexdigest()
    
    if path in _stat_cache_time and (current_time - _stat_cache_time[path]) < 1.0:
        stat = _stat_cache[path]
    else:
        try:
            stat = os.stat(path)
            _stat_cache[path] = stat
            _stat_cache_time[path] = current_time
        except Exception:
            # On error, use path only (no stat caching for errors)
            key = path
            return hashlib.sha256(key.encode()).hexdigest()
    
    key = f"{path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.sha256(key.encode()).hexdigest()


class ThumbnailCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir

    def get_cache_path(self, path):
        key = _thumbnail_cache_key(path)
        return os.path.join(self.cache_dir, f"{key}.png")

    def get_thumbnail(self, path):
        if is_cache_disabled():
            return None
        cache_path = self.get_cache_path(path)
        if os.path.exists(cache_path):
            try:
                img = Image.open(cache_path)
                # PNG text chunks should be automatically loaded into img.info
                # Log to verify they're present
                if hasattr(img, "info") and img.info:
                    thumb_keys = [k for k in img.info.keys() if k.startswith("thumb_")]
                    if thumb_keys:
                        logging.debug(f"[CACHE] Loaded thumbnail from cache: {path}, info keys with 'thumb_': {thumb_keys}")
                    else:
                        logging.warning(f"[CACHE] Loaded thumbnail from cache but no 'thumb_' keys in info: {path}, info keys: {list(img.info.keys())}")
                return img
            except Exception:
                return None
        return None

    def set_thumbnail(self, path, image):
        if is_cache_disabled():
            return
        cache_path = self.get_cache_path(path)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            pnginfo = PngImagePlugin.PngInfo()
            if hasattr(image, "info"):
                for key, value in image.info.items():
                    if isinstance(value, str):
                        pnginfo.add_text(key, value)
            image.save(cache_path, format="PNG", pnginfo=pnginfo)
        except Exception as e:
            logging.warning(f"Failed to save thumbnail to cache: {e}")
