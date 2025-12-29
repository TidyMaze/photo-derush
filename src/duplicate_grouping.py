"""Duplicate grouping for StratifiedGroupKFold cross-validation.

Groups images by perceptual hash similarity to prevent data leakage
from correlated photos (bursts, near-duplicates, same session).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from typing import Callable

try:
    import imagehash
    from PIL import Image

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logging.warning("[duplicate_grouping] imagehash not available, falling back to filename-based grouping")


def _get_cache_path(image_dir: str) -> str:
    """Get cache file path for duplicate groups."""
    cache_dir = os.path.expanduser("~/.photo-derush-cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Create hash of image_dir to use as cache key
    dir_hash = hashlib.md5(image_dir.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"duplicate_groups_{dir_hash}.pkl")


def _load_cached_groups(filenames: list[str], image_dir: str, cache_path: str) -> list[int] | None:
    """Load cached groups if available and still valid.
    
    Validates cache by checking:
    1. Exact filename match
    2. File modification times (mtimes) match
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        
        # Check exact filename match
        cached_filenames = cached.get("filenames", [])
        if cached_filenames != filenames:
            logging.debug(f"[duplicate_grouping] Cache invalid: filename list changed")
            return None
        
        # Check mtimes match
        cached_mtimes = cached.get("mtimes", {})
        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            try:
                current_mtime = os.path.getmtime(img_path)
                cached_mtime = cached_mtimes.get(fname)
                if cached_mtime is None or cached_mtime != current_mtime:
                    logging.debug(f"[duplicate_grouping] Cache invalid: mtime changed for {fname}")
                    return None
            except OSError:
                # File doesn't exist or can't be accessed
                logging.debug(f"[duplicate_grouping] Cache invalid: file not accessible {fname}")
                return None
        
        logging.info(f"[duplicate_grouping] Using cached groups ({len(set(cached['groups']))} groups)")
        return cached["groups"]
    except Exception as e:
        logging.debug(f"[duplicate_grouping] Failed to load cache: {e}")
    return None


def _save_cached_groups(filenames: list[str], groups: list[int], image_dir: str, cache_path: str):
    """Save groups to cache along with filenames and mtimes."""
    try:
        # Collect mtimes for all files
        mtimes: dict[str, float] = {}
        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            try:
                mtimes[fname] = os.path.getmtime(img_path)
            except OSError:
                # File doesn't exist or can't be accessed - skip it
                logging.debug(f"[duplicate_grouping] Could not get mtime for {fname}, skipping from cache")
        
        with open(cache_path, "wb") as f:
            pickle.dump({"filenames": filenames, "groups": groups, "mtimes": mtimes}, f)
    except Exception as e:
        logging.debug(f"[duplicate_grouping] Failed to save cache: {e}")


def create_duplicate_groups(
    filenames: list[str],
    image_dir: str,
    hamming_threshold: int = 5,
    fallback_group_fn: Callable[[str], str] | None = None,
    use_cache: bool = True,
) -> list[int]:
    """Group images by perceptual hash similarity.
    
    Args:
        filenames: List of image filenames
        image_dir: Directory containing images
        hamming_threshold: Maximum Hamming distance for grouping (default: 5)
        fallback_group_fn: Optional function to generate group ID from filename if imagehash fails
        use_cache: If True, use disk cache for groups (default: True)
        
    Returns:
        List of group IDs (one per filename)
    """
    start_time = time.perf_counter()
    
    # For very small datasets, skip grouping (not worth the overhead)
    if len(filenames) < 20:
        logging.debug(f"[duplicate_grouping] Skipping grouping for small dataset ({len(filenames)} images)")
        return list(range(len(filenames)))
    
    # Try cache first
    if use_cache:
        cache_path = _get_cache_path(image_dir)
        cached_groups = _load_cached_groups(filenames, image_dir, cache_path)
        if cached_groups is not None:
            elapsed = time.perf_counter() - start_time
            logging.info(f"[duplicate_grouping] Loaded from cache in {elapsed:.2f}s")
            return cached_groups
    
    if not IMAGEHASH_AVAILABLE:
        logging.warning("[duplicate_grouping] imagehash not available, using fallback grouping")
        if fallback_group_fn:
            groups = [hash(fallback_group_fn(f)) % 10000 for f in filenames]
        else:
            # Fallback: group by filename prefix (for bursts like IMG_001.jpg, IMG_002.jpg)
            groups = [hash(os.path.basename(f).rsplit("_", 1)[0] if "_" in os.path.basename(f) else f) % 10000 for f in filenames]
        if use_cache:
            _save_cached_groups(filenames, groups, image_dir, cache_path)
        return groups
    
    # OPTIMIZED: Open each image only once, compute hash and map in single pass
    hash_to_group: dict[str, int] = {}
    filename_to_hash: dict[str, str] = {}
    group_id = 0
    
    # First pass: compute all hashes
    for fname in filenames:
        img_path = os.path.join(image_dir, fname)
        try:
            with Image.open(img_path) as img:
                phash = imagehash.phash(img)
                phash_str = str(phash)
                filename_to_hash[fname] = phash_str
        except Exception as e:
            logging.debug(f"[duplicate_grouping] Failed to hash {fname}: {e}")
            # Use fallback for this file
            if fallback_group_fn:
                phash_str = f"fallback_{hash(fallback_group_fn(fname)) % 10000}"
            else:
                phash_str = f"fallback_{fname}"
            filename_to_hash[fname] = phash_str
    
    # Second pass: assign groups (check similarity)
    for fname, phash_str in filename_to_hash.items():
        if phash_str.startswith("fallback_"):
            # Fallback: assign unique group or use existing
            if phash_str not in hash_to_group:
                hash_to_group[phash_str] = group_id
                group_id += 1
            continue
        
        # Find existing group with similar hash
        assigned = False
        phash = imagehash.hex_to_hash(phash_str)
        for existing_hash_str, gid in hash_to_group.items():
            if existing_hash_str.startswith("fallback_"):
                continue
            existing_hash = imagehash.hex_to_hash(existing_hash_str)
            if phash - existing_hash <= hamming_threshold:
                hash_to_group[phash_str] = gid
                assigned = True
                break
        
        if not assigned:
            hash_to_group[phash_str] = group_id
            group_id += 1
    
    # Map filenames to group IDs
    result = [hash_to_group.get(filename_to_hash[fname], group_id) for fname in filenames]
    
    elapsed = time.perf_counter() - start_time
    logging.info(f"[duplicate_grouping] Created {len(set(result))} groups from {len(filenames)} images in {elapsed:.2f}s")
    
    # Save to cache
    if use_cache:
        _save_cached_groups(filenames, result, image_dir, cache_path)
    
    return result

