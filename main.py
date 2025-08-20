import os
from PIL import Image
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS  # Pillow >=9.1.0
except AttributeError:
    LANCZOS_RESAMPLE = Image.LANCZOS  # type: ignore[attr-defined]

import imagehash  # type: ignore
import faiss  # type: ignore
import numpy as np  # type: ignore
import cv2  # type: ignore
import logging
from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt, open_full_image_qt
from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt_async
from photo_derush.image_manager import image_manager
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from precompute import compute_duplicate_groups
import threading

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

def list_images(directory):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in exts]

def list_extensions(directory):
    return sorted(set(os.path.splitext(f)[1].lower() for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f)) and '.' in f))

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

def compute_dhash(image_path):
    img = image_manager.get_image(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image for dhash: {image_path}")
    try:
        img.thumbnail((32, 32), LANCZOS_RESAMPLE)
    except Exception:
        pass
    return imagehash.dhash(img)

def hash_one(img_name, directory):
    import time
    start = time.time()
    img_path = os.path.join(directory, img_name)
    try:
        dh = compute_dhash(img_path)
        h_bytes = int(str(dh), 16).to_bytes(8, 'big')
        hash_arr = np.frombuffer(h_bytes, dtype='uint8')
        elapsed = time.time() - start
        logging.getLogger(__name__).debug(f"[Hashing] {img_name} hashed in {elapsed:.3f}s (thread={threading.current_thread().name})")
        return img_name, hash_arr, dh, None
    except Exception as e:
        elapsed = time.time() - start
        logging.getLogger(__name__).error(f"[Hashing] {img_name} failed in {elapsed:.3f}s: {e}")
        return img_name, None, None, e

def parallel_hash_images(image_paths, directory, max_workers=None, use_threads=False):
    """Hash images in parallel. Returns (valid_paths, hashes, image_hashes, errors)"""
    hashes = []
    valid_paths = []
    image_hashes = {}
    errors = {}
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(hash_one, img_name, directory): img_name for img_name in image_paths}
        for i, future in enumerate(as_completed(future_to_img)):
            img_name, hash_arr, dh, err = future.result()
            if hash_arr is not None:
                hashes.append(hash_arr)
                valid_paths.append(img_name)
                image_hashes[img_name] = hash_arr
            else:
                errors[img_name] = err
    return valid_paths, hashes, image_hashes, errors

def group_hashes(hashes, valid_paths, hamming_thresh=5):
    """Group hashes using faiss. Returns clusters (list of lists of image names)."""
    if not hashes:
        return []
    # Ensure hashes are np.uint8 and shape (n, 8) for 64 bits
    hashes_np = np.stack(hashes)
    if hashes_np.dtype != np.uint8:
        hashes_np = hashes_np.astype(np.uint8)
    if hashes_np.shape[1] != 8:
        raise ValueError(f"Expected hash shape (n, 8), got {hashes_np.shape}")
    index = faiss.IndexBinaryFlat(64)  # type: ignore
    index.add(hashes_np)  # type: ignore
    clusters = []
    visited = set()
    for i, h in enumerate(hashes_np):
        if i in visited:
            continue
        query = h[np.newaxis, :].astype(np.uint8)
        lims, D, I = index.range_search(query, hamming_thresh)  # type: ignore
        cluster = [valid_paths[j] for j in I[lims[0]:lims[1]] if j not in visited]
        for j in I[lims[0]:lims[1]]:
            visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters

def cluster_duplicates(image_paths, directory, hamming_thresh=5):
    valid_paths, hashes, image_hashes, errors = parallel_hash_images(image_paths, directory)
    for i, img_name in enumerate(valid_paths):
        if i % 20 == 0 or i == len(valid_paths) - 1:
            logging.info(f"[Hashing] (parallel) {i+1}/{len(image_paths)} hashed")
    for img_name, err in errors.items():
        logging.warning(f"[Hashing] Failed to hash {img_name}: {err}")
    logging.info(f"Computed {len(hashes)} hashes for {len(image_paths)} images.")
    clusters = group_hashes(hashes, valid_paths, hamming_thresh)
    logging.info(f"Found {len(clusters)} clusters with Hamming threshold {hamming_thresh}.")
    return clusters, image_hashes

def main_duplicate_detection(clusters=None, image_hashes=None):
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    if clusters is None or image_hashes is None:
        images = list_images(directory)
        clusters, image_hashes = cluster_duplicates(images, directory)
    print(f"Found {len(clusters)} duplicate clusters.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}: {cluster}")

def duplicate_slayer(image_dir, trash_dir, show_ui=True):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"[Duplicate Slayer] Found {len(images)} images in {image_dir}.")
    if not images:
        print("[Duplicate Slayer] No images found. Exiting.")
        return [], []
    trashed = []
    kept = [os.path.join(image_dir, images[0])]
    for img in images[1:]:
        src = os.path.join(image_dir, img)
        dst = os.path.join(trash_dir, img)
        print(f"[Duplicate Slayer] Moving duplicate: {img} -> trash.")
        os.rename(src, dst)
        trashed.append(img)
    all_images = [images[0]] + trashed
    # Skip UI during automated tests
    running_tests = os.environ.get("PYTEST_CURRENT_TEST") is not None
    if show_ui and not running_tests:
        print(f"[Duplicate Slayer] Loading {len(all_images[:MAX_IMAGES])} images in UI.")
        show_lightroom_ui(all_images[:MAX_IMAGES], image_dir, trashed_paths=trashed[:MAX_IMAGES-1], trashed_dir=trash_dir)
    return kept, [os.path.join(trash_dir, t) for t in trashed]

def prepare_images_and_groups(directory: str, max_images: int = MAX_IMAGES):
    images = list_images(directory)
    logging.info("[Prep] Found %d images in %s", len(images), directory)
    if not images:
        return [], {}, {"total_images": 0, "duplicate_group_count": 0, "duration_seconds": 0.0}
    logging.info("[Hashing] Starting hash + group computation for %d images", len(images))
    start_hash_time = time.perf_counter()
    valid_paths, hashes, image_hashes, errors = parallel_hash_images(images, directory)
    for img_name, err in errors.items():
        logging.warning("[Hashing] Failed for %s: %s", img_name, err)
    group_ids, group_cardinality, hash_map = compute_duplicate_groups(hashes)
    duration = time.perf_counter() - start_hash_time
    duplicate_group_count = len([g for g in set(group_ids.values()) if g is not None])
    logging.info("[Hashing] Finished hash + group computation in %.2fs. Duplicate groups: %d", duration, duplicate_group_count)
    image_info = {}
    for idx, img in enumerate(valid_paths):
        image_info[img] = {"hash": hash_map.get(idx), "group": group_ids.get(idx)}
    stats = {"total_images": len(valid_paths), "duplicate_group_count": duplicate_group_count, "duration_seconds": duration}
    return valid_paths, image_info, stats

def main():
    directory = '/Users/yannrolland/Pictures/photo-dataset'
    print("Welcome to Photo Derush Script (async mode)!")
    # Launch UI immediately; heavy preprocessing happens in background thread.
    show_lightroom_ui_qt_async(directory, MAX_IMAGES)

if __name__ == "__main__":
    import sys
    if not any(x in sys.argv[0] for x in ["pytest", "test_", "_test"]):
        main()
