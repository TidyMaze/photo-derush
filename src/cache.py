import os
import hashlib
from PIL import Image

CACHE_DIR = os.path.expanduser('~/.photo-derush-cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def _thumbnail_cache_key(path):
    try:
        stat = os.stat(path)
        key = f"{path}:{stat.st_mtime}:{stat.st_size}"
    except Exception:
        key = path
    return hashlib.sha256(key.encode()).hexdigest()

class ThumbnailCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir

    def get_cache_path(self, path):
        key = _thumbnail_cache_key(path)
        return os.path.join(self.cache_dir, f"{key}.png")

    def get_thumbnail(self, path):
        cache_path = self.get_cache_path(path)
        if os.path.exists(cache_path):
            try:
                return Image.open(cache_path)
            except Exception:
                return None
        return None

    def set_thumbnail(self, path, image):
        cache_path = self.get_cache_path(path)
        try:
            image.save(cache_path, format="PNG")
        except Exception:
            pass

