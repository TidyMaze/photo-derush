print("[Startup] main.py started (print statement)")
import os
import threading
import numpy as np
import cv2
import logging
from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt, open_full_image_qt
from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt_async
from photo_derush.image_manager import image_manager
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from photo_derush.duplicate import cluster_duplicates

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_IMAGES = 400

# Added for tests expecting this helper
_MAX_THUMB_MEM = 500
_thumbnail_mem_cache: "OrderedDict[tuple, Image.Image]" = OrderedDict()

def _memoize_thumb(mem_key, img):
    # Promote existing key
    if mem_key in _thumbnail_mem_cache:
        _thumbnail_mem_cache.move_to_end(mem_key)
    else:
        _thumbnail_mem_cache[mem_key] = img
    # Evict oldest if over capacity
    while len(_thumbnail_mem_cache) > _MAX_THUMB_MEM:
        _thumbnail_mem_cache.popitem(last=False)

def cache_thumbnail(src_path: str, thumb_path: str):
    """Return a PIL image object for thumbnail and whether it was loaded from cache.
    If disk thumbnail exists and is up-to-date, load it. Otherwise create via ImageManager and save.
    Also memoize in-process to avoid repeated disk hits within a run.
    """
    logger = logging.getLogger(__name__)
    try:
        mtime = os.path.getmtime(src_path)
    except OSError:
        logger.warning("[cache_thumbnail] Source missing: %s", src_path)
        return None, False
    mem_key = (src_path, mtime)
    if mem_key in _thumbnail_mem_cache:
        # Promote to recent
        img_mem = _thumbnail_mem_cache[mem_key]
        _memoize_thumb(mem_key, img_mem)
        logger.info("Loaded cached thumbnail for %s", src_path)
        return img_mem, True
    # Ensure dir
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    disk_ok = False
    if os.path.exists(thumb_path):
        try:
            if os.path.getmtime(thumb_path) >= mtime:
                from PIL import Image
                with Image.open(thumb_path) as disk_img:
                    disk_img.load()
                    img = disk_img.copy()  # detach from file descriptor
                _memoize_thumb(mem_key, img)
                logger.info("Loaded cached thumbnail for %s", src_path)
                return img, True
        except Exception as e:  # noqa: PERF203
            logger.info("[cache_thumbnail] Failed loading disk cache %s: %s (will recreate)", thumb_path, e)
    # Create new thumbnail using ImageManager (which also persists)
    thumb_img = image_manager.get_thumbnail(src_path, (200, 200))
    if thumb_img is not None:
        try:
            thumb_img.save(thumb_path)
        except Exception as e:  # noqa: PERF203
            logger.info("[cache_thumbnail] Could not save thumbnail %s: %s", thumb_path, e)
        _memoize_thumb(mem_key, thumb_img)
        logger.info("Created and cached thumbnail for %s", src_path)
        return thumb_img, False
    logger.warning("[cache_thumbnail] Could not create thumbnail for %s", src_path)
    return None, False

def is_image_extension(ext):
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def open_full_image(img_path):
    open_full_image_qt(img_path)

def compute_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_sharpness_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = {}
    features['variance_laplacian'] = cv2.Laplacian(img, cv2.CV_64F).var()
    features['tenengrad'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Tenengrad
    features['brenner'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Brenner
    features['wavelet_energy'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Wavelet energy
    return features

def show_lightroom_ui(image_paths, directory, trashed_paths=None, trashed_dir=None, on_window_opened=None):
    show_lightroom_ui_qt(image_paths, directory, trashed_paths, trashed_dir, on_window_opened=on_window_opened)

def main_duplicate_detection(clusters=None, image_hashes=None):
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    from photo_derush.duplicate import cluster_duplicates
    import os
    def list_images(directory):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        return [f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]
    if clusters is None or image_hashes is None:
        images = list_images(directory)
        clusters, image_hashes = cluster_duplicates(images, directory)
    print(f"Found {len(clusters)} duplicate clusters.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")

def duplicate_slayer(src_dir, trash_dir):
    """
    Remove duplicate files (by content) in src_dir, move duplicates to trash_dir, keep one copy.
    Returns (kept, trashed): lists of file paths.
    """
    import hashlib
    kept = []
    trashed = []
    hashes = {}
    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash not in hashes:
            hashes[file_hash] = fpath
            kept.append(fpath)
        else:
            # Move duplicate to trash
            dest = os.path.join(trash_dir, fname)
            os.rename(fpath, dest)
            trashed.append(fpath)
    return kept, trashed

def main():
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    print("Welcome to Photo Derush Script (async mode)!")
    # Launch UI immediately; heavy preprocessing happens in background thread.
    show_lightroom_ui_qt_async(directory, MAX_IMAGES)

if __name__ == "__main__":
    import sys
    if not any(x in sys.argv[0] for x in ["pytest", "test_", "_test"]):
        main()
