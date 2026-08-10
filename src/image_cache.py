"""
Shared PIL Image cache - lightweight wrapper to avoid caching full images in memory.

This module provides convenience functions for opening images without caching
full decoded images in memory. The OS file system cache handles repeated opens efficiently.

Memory impact: Removed ~1.9GB of in-memory image cache (100 images × 19MB each).
Performance: OS file system cache provides similar benefits with minimal memory cost.
"""

import contextlib
import logging
import os
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


RAW_EXTS = {".arw", ".cr2", ".nef", ".dng", ".orf", ".rw2", ".pef", ".srw"}
RASTER_EXTS = [".JPG", ".jpg", ".jpeg", ".JPEG"]


def normalize_path(path: str) -> str:
    """
    Normalize file path by resolving non-breaking spaces (\xa0) vs regular spaces (U+0020).
    Handles path encoding mismatches between Darktable, Lua, and Windows OS filesystem.
    """
    if not path or not isinstance(path, str):
        return path
    if os.path.exists(path):
        return path

    norm_p = os.path.normpath(path)
    parts = norm_p.split(os.sep)
    if not parts:
        return path

    if parts[0].endswith(':'):
        cur = parts[0] + os.sep
        start_idx = 1
    else:
        cur = ''
        start_idx = 0

    for part in parts[start_idx:]:
        if not part:
            continue
        cand = os.path.join(cur, part)
        if os.path.exists(cand):
            cur = cand
            continue

        alt_nbsp = os.path.join(cur, part.replace(' ', '\xa0'))
        if os.path.exists(alt_nbsp):
            cur = alt_nbsp
            continue

        alt_space = os.path.join(cur, part.replace('\xa0', ' '))
        if os.path.exists(alt_space):
            cur = alt_space
            continue

        cur = cand

    return cur


def find_paired_jpg(raw_path: str) -> Optional[str]:
    """Find rastered JPG version for a RAW file if it exists in the same directory."""
    if not raw_path:
        return None
    raw_path = normalize_path(raw_path)
    stem, ext = os.path.splitext(raw_path)
    if ext.lower() in RAW_EXTS:
        for r_ext in RASTER_EXTS:
            candidate = stem + r_ext
            if os.path.isfile(candidate):
                return candidate
            # Also check darktable_exported subfolder
            parent, fname = os.path.split(stem)
            exp_candidate = os.path.join(parent, "darktable_exported", fname + r_ext)
            if os.path.isfile(exp_candidate):
                return exp_candidate
    return None


@contextlib.contextmanager
def get_cached_image(path: str):
    """
    Open an image file with a context manager to ensure it is closed and locks are released.
    Prefers rastered JPG version (.JPG/.jpg) if path is RAW (.ARW).
    Includes fallback for RAW formats by extracting largest embedded JPEG preview if no JPG file exists.
    """
    if not path or path.lower().endswith((".xmp", ".dop", ".xml", ".json", ".txt", ".db", ".joblib", ".pkl")):
        yield None
        return
    path = normalize_path(path)
    paired_jpg = find_paired_jpg(path)
    actual_path = paired_jpg if paired_jpg else path
    img = None
    try:
        img = Image.open(actual_path)
        yield img
    except Exception as e:
        # Fallback for RAW files: extract largest embedded JPEG preview
        ext = os.path.splitext(actual_path)[1].lower()
        if ext in RAW_EXTS:
            try:
                import io
                with open(actual_path, "rb") as f:
                    data = f.read()

                largest_img = None
                largest_area = 0
                pos = 0
                while True:
                    idx = data.find(b"\xff\xd8\xff", pos)
                    if idx == -1:
                        break
                    end_idx = data.find(b"\xff\xd9", idx)
                    if end_idx != -1:
                        try:
                            candidate = Image.open(io.BytesIO(data[idx : end_idx + 2]))
                            area = candidate.size[0] * candidate.size[1]
                            if area > largest_area:
                                largest_area = area
                                largest_img = candidate
                        except Exception:
                            pass
                    pos = idx + 1

                if largest_img is not None:
                    yield largest_img
                    return
            except Exception as ex:
                logger.warning(f"RAW preview extraction error for {actual_path}: {ex}")

        logger.warning(f"Failed to open image {actual_path}: {e}")
        yield None
    finally:
        if img is not None:
            img.close()


@contextlib.contextmanager
def get_cached_image_for_exif(path: str):
    """
    Open an image file for EXIF extraction with a context manager to ensure locks are released.
    Prefers rastered JPG version (.JPG/.jpg) if path is RAW (.ARW).
    """
    if not path:
        yield None
        return
    path = normalize_path(path)
    paired_jpg = find_paired_jpg(path)
    actual_path = paired_jpg if paired_jpg else path
    img = None
    try:
        img = Image.open(actual_path)
        yield img
    except Exception as e:
        logger.warning(f"Failed to open image for EXIF {actual_path}: {e}")
        yield None
    finally:
        if img is not None:
            img.close()


def clear_image_cache():
    """
    Clear image cache (no-op, kept for API compatibility).
    """
    pass

