"""Feature extraction & cache management (SRP).
Provides: FEATURE_COUNT, USE_FULL_FEATURES, extract_features, batch_extract_features,
load_feature_cache, save_feature_cache, safe_initialize_feature_cache.
"""

from __future__ import annotations

import atexit
import logging
import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageStat

from src.feature_data import ImagePreprocessResult

if TYPE_CHECKING:
    from multiprocessing import Pool

FEATURE_CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feature_cache.pkl"))
FEATURE_CACHE_PATH = os.environ.get("FEATURE_CACHE_PATH", FEATURE_CACHE_PATH)
USE_FULL_FEATURES = os.environ.get("FULL_FEATURES", "0") == "1"
# FEATURE_COUNT: 71 core features (FAST) with real EXIF extraction
# FAST: 71 features | FULL: 95 features (16 additional histogram bins + 8 advanced features)
# With new photography features: +6 features = 77 (FAST) or 101 (FULL)
# With person detection: +1 feature = 78 (FAST) or 102 (FULL)
FEATURE_COUNT = 102 if USE_FULL_FEATURES else 78
if not globals().get("_FEATURE_MODE_LOGGED"):
    logging.info(f"[features] Mode: {'FULL' if USE_FULL_FEATURES else 'FAST'} (FEATURE_COUNT={FEATURE_COUNT})")
    _FEATURE_MODE_LOGGED = True
feature_cache: dict | None = None
# Module-level cache for object detection to avoid repeated loads during feature extraction
_object_detection_cache: dict | None = None
_object_detection_cache_mtime: float = 0
# Track active pool to prevent multiple concurrent pools
_active_pool: "Pool | None" = None
_pool_lock = None  # Will be initialized as threading.Lock when needed

def _cleanup_pools():
    """Cleanup function registered with atexit to ensure pools are terminated."""
    global _active_pool
    if _active_pool is not None:
        try:
            logging.info("[features] Cleaning up multiprocessing pool on exit")
            try:
                _active_pool.terminate()
            except Exception:
                pass  # Pool might already be terminated
            try:
                _active_pool.join(timeout=1.0)
            except Exception:
                pass  # Join might fail if pool is already closed
            # Force kill if still alive
            try:
                if hasattr(_active_pool, '_processes') and _active_pool._processes:
                    for p in _active_pool._processes:
                        try:
                            if p is not None and p.is_alive():
                                p.terminate()
                                p.join(timeout=0.5)
                                if p.is_alive():
                                    p.kill()
                        except Exception:
                            pass
            except Exception:
                pass  # Process list might not be accessible
        except Exception as e:
            logging.warning(f"[features] Error cleaning up pool: {e}")
        finally:
            _active_pool = None

# Register cleanup on exit
atexit.register(_cleanup_pools)


def _migrate_cache_keys(cache: dict) -> tuple[dict, bool]:
    """Migrate cache keys to absolute paths."""
    migrated = {os.path.abspath(k): v for k, v in cache.items()}
    changed = len(migrated) != len(cache) or any(k != os.path.abspath(k) for k in cache.keys())
    return migrated, changed


def _validate_cache_entries(cache: dict) -> tuple[dict, int]:
    """Remove invalid cache entries.

    Checks:
    1. Entry structure: (list|tuple) of FEATURE_COUNT floats
    2. File existence: key path must exist
    3. File staleness: skip if mtime > 5 minutes old (optional, warns)
    """
    valid = {}
    removed = 0
    for path, feats in cache.items():
        # Validate structure
        if not isinstance(feats, (list, tuple)) or len(feats) != FEATURE_COUNT:
            removed += 1
            continue
        # Validate file still exists
        if not os.path.isfile(path):
            logging.debug(f"[features] Removed cache entry for deleted file: {path}")
            removed += 1
            continue
        # Optional: warn if file was modified since extraction (helps detect accidental overwrites)
        try:
            mtime = os.path.getmtime(path)
            import time

            age_sec = time.time() - mtime
            if age_sec < 0:
                logging.warning(f"[features] File mtime in future (clock skew?): {path}")
                removed += 1
                continue
        except (OSError, Exception) as e:
            logging.exception("[features] Could not check mtime for %s: %s", path, e)
            removed += 1
            continue
        # Valid entry
        valid[path] = feats
    return valid, removed


def load_feature_cache() -> dict:
    """Load feature cache with automatic version/mode detection.

    Automatically purges cache if:
    - Feature count changed (FAST/FULL mode switch)
    - Cache corrupted or invalid entries
    - File not found

    Returns: {path: [features]} dict or {} if empty/invalid
    """
    if not os.path.exists(FEATURE_CACHE_PATH):
        return {}

    try:
        import time
        cache_mtime = os.path.getmtime(FEATURE_CACHE_PATH)
        cache_age = time.time() - cache_mtime
        
        with open(FEATURE_CACHE_PATH, "rb") as f:
            raw = pickle.load(f)

        # Check for metadata (v2 format with header)
        if isinstance(raw, dict) and "__metadata__" in raw:
            metadata = raw.pop("__metadata__")
            cache = raw
            cached_feature_count = metadata.get("feature_count", 0)
            if cached_feature_count != FEATURE_COUNT:
                logging.warning(
                    f"[features] Cache feature_count mismatch: cached={cached_feature_count} current={FEATURE_COUNT}; deleting stale cache"
                )
                try:
                    os.remove(FEATURE_CACHE_PATH)
                except Exception as e:
                    logging.error(f"[features] Failed deleting stale cache: {e}")
                return {}
        else:
            # v1 format (no metadata); assume it's valid
            cache = raw

        cache, key_changed = _migrate_cache_keys(cache)
        if key_changed:
            logging.info(f"[features] Migrated cache keys to absolute paths ({len(cache)})")

        # Skip expensive validation if cache is recent (< 1 hour old) and no key migration needed
        # This significantly speeds up cache loading for large caches
        if cache_age < 3600 and not key_changed:
            # Quick validation: only check structure, skip file existence/mtime checks
            valid = {}
            removed = 0
            for path, feats in cache.items():
                if isinstance(feats, (list, tuple)) and len(feats) == FEATURE_COUNT:
                    valid[path] = feats
                else:
                    removed += 1
            cache = valid
            if removed:
                logging.debug(f"[features] Quick validation removed {removed} invalid entries")
        else:
            # Full validation for older caches or after migration
            cache, removed = _validate_cache_entries(cache)
            if removed:
                logging.info(f"[features] Purged {removed} stale entries (expected {FEATURE_COUNT})")

        if key_changed or removed:
            save_feature_cache(cache)

        logging.info(
            f"[features] Cache loaded: {len(cache)} valid entries (mode={'FULL' if USE_FULL_FEATURES else 'FAST'})"
        )
        return cache
    except pickle.UnpicklingError as e:
        logging.error(f"[features] Cache corruption detected: {e}; resetting cache")
        try:
            os.remove(FEATURE_CACHE_PATH)
        except Exception:
            logging.exception("Error computing feature chunk")
            raise
        return {}
    except Exception as e:
        logging.warning(f"[features] Failed to load feature cache: {e}")
        return {}


def save_feature_cache(cache: dict):
    """Save feature cache with metadata header for version detection."""
    try:
        # Add metadata header for v2 format
        data = {
            "__metadata__": {
                "feature_count": FEATURE_COUNT,
                "mode": "FULL" if USE_FULL_FEATURES else "FAST",
                "version": 2,
            }
        }
        data.update(cache)

        # Atomic write: write to temp, then rename
        tmp_path = f"{FEATURE_CACHE_PATH}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, FEATURE_CACHE_PATH)
        logging.info(f"[features] Cache saved ({len(cache)} entries, mode={'FULL' if USE_FULL_FEATURES else 'FAST'})")
    except Exception as e:
        logging.error(f"[features] Failed saving cache: {e}")


def _extract_exif_value(exif_data: dict, tag: int, default: float, is_ratio: bool = False) -> float:
    """Extract and convert EXIF value, handling ratios and errors. Optimized for speed."""
    try:
        value = exif_data.get(tag)  # Use .get() to avoid KeyError check
        if value is None:
            return default
        if is_ratio and isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1]) if value[1] != 0 else default
        return float(value)
    except (ValueError, TypeError, ZeroDivisionError, KeyError):
        return default


def _extract_exif_data(img: Image.Image, path: str) -> dict:
    """Extract EXIF metadata from image. Optimized for speed."""
    # Default values (returned if no EXIF or extraction fails)
    result = {
        "iso": 400.0,
        "aperture": 2.8,
        "shutter_speed": 0.001,
        "flash_fired": 0.0,
        "focal_length_35mm": 50.0,
        "digital_zoom": 1.0,
        "exposure_compensation": 0.0,
        "white_balance": 0.0,
        "exposure_mode": 0.0,
        "metering_mode": 5.0,
        "datetime_str": None,
    }

    try:
        # Try _getexif() first (direct dict access, faster), fallback to getexif()
        exif_data = None
        if hasattr(img, "_getexif") and callable(img._getexif):
            try:
                exif_data = img._getexif()
            except Exception:
                pass
        
        if not exif_data and hasattr(img, "getexif") and callable(img.getexif):
            try:
                exif_obj = img.getexif()
                if exif_obj:
                    exif_data = dict(exif_obj)  # Convert to dict for compatibility
            except Exception:
                pass
        
        if not exif_data or not isinstance(exif_data, dict):
            return result  # Early return if no EXIF (avoids dict lookups)

        # Optimize: cache dict lookups, use direct access where possible
        # ISO (34855)
        if 34855 in exif_data:
            result["iso"] = _extract_exif_value(exif_data, 34855, 400.0)
        
        # Aperture (33437)
        if 33437 in exif_data:
            result["aperture"] = _extract_exif_value(exif_data, 33437, 2.8, is_ratio=True)
        
        # Shutter speed (33434)
        if 33434 in exif_data:
            result["shutter_speed"] = _extract_exif_value(exif_data, 33434, 0.001, is_ratio=True)

        # Flash (37385) - check bit 0
        if 37385 in exif_data:
            try:
                result["flash_fired"] = float(1 if (int(exif_data[37385]) & 0x01) else 0)
            except (ValueError, TypeError):
                pass

        # Focal length (37386) - prefer 35mm equivalent (41989)
        if 41989 in exif_data:
            result["focal_length_35mm"] = _extract_exif_value(exif_data, 41989, 50.0)
        elif 37386 in exif_data:
            result["focal_length_35mm"] = _extract_exif_value(exif_data, 37386, 50.0, is_ratio=True)

        # Digital zoom (41988)
        if 41988 in exif_data:
            result["digital_zoom"] = _extract_exif_value(exif_data, 41988, 1.0, is_ratio=True)
        
        # Exposure compensation (37380)
        if 37380 in exif_data:
            result["exposure_compensation"] = _extract_exif_value(exif_data, 37380, 0.0, is_ratio=True)
        
        # White balance (37384)
        if 37384 in exif_data:
            result["white_balance"] = _extract_exif_value(exif_data, 37384, 0.0)
        
        # Exposure mode (41986)
        if 41986 in exif_data:
            result["exposure_mode"] = _extract_exif_value(exif_data, 41986, 0.0)
        
        # Metering mode (37383)
        if 37383 in exif_data:
            result["metering_mode"] = _extract_exif_value(exif_data, 37383, 5.0)

        # Extract datetime - DateTimeOriginal (36867) or DateTime (306)
        datetime_str = exif_data.get(36867) or exif_data.get(306)
        if datetime_str:
            result["datetime_str"] = str(datetime_str)  # type: ignore[assignment]

        return result
    except Exception as e:
        logging.debug(f"[features] EXIF extraction failed for {path}: {e}")
        return result


def _compute_histograms(arr: np.ndarray, bins: int) -> np.ndarray:
    """Compute normalized RGB histograms.
    
    Optimized: use bincount with direct integer division (faster than float division).
    """
    hists = []
    # Convert to uint8 for faster bincount
    arr_uint8 = arr.astype(np.uint8)
    # Use integer division: bin = pixel_value * bins // 256 (faster than float division)
    bin_scale = bins * 256  # Pre-compute to avoid repeated multiplication
    for ch in range(3):
        # Direct integer division is faster than float division + clip
        channel = arr_uint8[:, :, ch]
        # Compute bin indices: pixel * bins // 256, then clip to [0, bins-1]
        bin_indices = (channel.astype(np.uint32) * bins) >> 8  # Right shift by 8 = divide by 256
        bin_indices = np.clip(bin_indices, 0, bins - 1).astype(np.int32)
        # Count occurrences in each bin (bincount is very fast)
        hist = np.bincount(bin_indices.flatten(), minlength=bins).astype(np.float32)
        # Normalize
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        hists.append(hist)
    return np.concatenate(hists)


def _preprocess_image(path: str) -> ImagePreprocessResult | None:
    import time
    prep_start = time.perf_counter()
    basename = os.path.basename(path)
    try:
        # Open image with lazy loading - don't decode until needed
        open_start = time.perf_counter()
        # Use verify=False for faster loading (skip format verification)
        img_original = Image.open(path)
        # Extract EXIF BEFORE resizing (EXIF data is lost after resize)
        # Also extract BEFORE load() to avoid decoding if EXIF extraction fails early
        exif_start = time.perf_counter()
        try:
            exif_data = _extract_exif_data(img_original, path)
        except Exception as e:
            logging.warning(f"[features] EXIF extraction error for {path}: {e}")
            exif_data = {
                "iso": 400.0, "aperture": 2.8, "shutter_speed": 0.001, "flash_fired": 0.0,
                "focal_length_35mm": 50.0, "digital_zoom": 1.0, "exposure_compensation": 0.0,
                "white_balance": 0.0, "exposure_mode": 0.0, "metering_mode": 5.0, "datetime_str": None,
            }
        exif_time = time.perf_counter() - exif_start
        
        # Now load and get dimensions (triggers decode)
        img_original.load()  # Force decode now
        w_orig, h_orig = img_original.size
        open_time = time.perf_counter() - open_start
        if open_time > 0.5:
            logging.info(f"[features] SLOW image open for {basename}: {open_time:.2f}s")
        
        # Downsample very large images early to speed up all subsequent operations
        # For images > 0.5MP, downsample to ~0.5MP (very aggressive for speed)
        # Use thumbnail() for very large images (faster than resize), LANCZOS otherwise
        max_pixels_prep = 512 * 1024  # 0.5MP - very aggressive downsampling for speed
        if w_orig * h_orig > max_pixels_prep:
            scale = np.sqrt(max_pixels_prep / (w_orig * h_orig))
            w, h = int(w_orig * scale), int(h_orig * scale)
            # Use thumbnail() for very large images (>4MP) - it's faster and modifies in-place
            if w_orig * h_orig > 4 * 1024 * 1024:
                img_original.thumbnail((w, h), Image.Resampling.NEAREST)
                w, h = img_original.size  # thumbnail may adjust aspect ratio
            else:
                img_original = img_original.resize((w, h), Image.Resampling.LANCZOS)
            logging.debug(f"[features] Downsampled {basename} from {w_orig}x{h_orig} to {w}x{h} for faster processing")
        else:
            w, h = w_orig, h_orig
        
        if exif_time > 0.5:
            logging.info(f"[features] SLOW EXIF extraction for {basename}: {exif_time:.2f}s")

        convert_start = time.perf_counter()
        img = img_original.convert("RGB")
        convert_time = time.perf_counter() - convert_start
        if convert_time > 0.5:
            logging.info(f"[features] SLOW RGB conversion for {basename}: {convert_time:.2f}s")

        # Optimize: compute stats and grayscale together to avoid double scan
        stat_start = time.perf_counter()
        gray = img.convert("L")
        gray_arr = np.array(gray, dtype=np.float32)
        # Compute RGB stats from numpy array (faster than ImageStat for downsampled images)
        arr = np.array(img)
        mean_r, mean_g, mean_b = float(arr[:, :, 0].mean()), float(arr[:, :, 1].mean()), float(arr[:, :, 2].mean())
        std_r, std_g, std_b = float(arr[:, :, 0].std()), float(arr[:, :, 1].std()), float(arr[:, :, 2].std())
        mean_brightness = float(gray_arr.mean())
        std_brightness = float(gray_arr.std())
        stat_time = time.perf_counter() - stat_start
        if stat_time > 0.5:
            logging.info(f"[features] SLOW stat computation for {basename}: {stat_time:.2f}s")

        hist_start = time.perf_counter()
        bins = 16 if USE_FULL_FEATURES else 8
        # Use already-computed arr from stats (no need to convert again)
        hist_feat = _compute_histograms(arr, bins)
        hist_time = time.perf_counter() - hist_start
        if hist_time > 0.5:
            logging.info(f"[features] SLOW histogram computation for {basename}: {hist_time:.2f}s")

        file_size = os.path.getsize(path)
        aspect = w / h if h else 0.0
        
        prep_total = time.perf_counter() - prep_start
        if prep_total > 1.0:
            logging.info(f"[features] SLOW preprocessing for {basename}: total={prep_total:.2f}s (open={open_time:.2f}s, exif={exif_time:.2f}s, convert={convert_time:.2f}s, stat={stat_time:.2f}s, hist={hist_time:.2f}s)")

        return ImagePreprocessResult(
            img=img,
            gray_arr=gray_arr,
            w=w,
            h=h,
            mean_r=mean_r,
            mean_g=mean_g,
            mean_b=mean_b,
            std_r=std_r,
            std_g=std_g,
            std_b=std_b,
            mean_brightness=mean_brightness,
            std_brightness=std_brightness,
            hist_feat=hist_feat,
            file_size=file_size,
            aspect=aspect,
            exif_data=exif_data,
        )
    except Exception as e:
        logging.exception("[features] _preprocess_image failed for %s: %s", path, e)
        return None


def _compute_edge_features(gray_arr: np.ndarray) -> tuple[float, float]:
    """Compute edge density and strength using Sobel operator."""
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Fallback: simple gradient
    grad_x = np.diff(gray_arr.astype(np.float32), axis=1)  # shape: (h, w-1)
    grad_y = np.diff(gray_arr.astype(np.float32), axis=0)  # shape: (h-1, w)
    # Match shapes: take overlapping region
    magnitude = np.sqrt(grad_x[:-1, :] ** 2 + grad_y[:, :-1] ** 2)
    edge_density = float(np.mean(magnitude > np.percentile(magnitude, 75))) if magnitude.size > 0 else 0.0
    edge_strength = float(np.mean(magnitude)) if magnitude.size > 0 else 0.0
    return edge_density, edge_strength


def _compute_corner_count(gray_arr: np.ndarray) -> float:
    """Estimate corner count using Harris corner detection approximation."""
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Fallback: count high-gradient points
    grad_x = np.diff(gray_arr.astype(np.float32), axis=1)  # shape: (h, w-1)
    grad_y = np.diff(gray_arr.astype(np.float32), axis=0)  # shape: (h-1, w)
    # Match shapes: take overlapping region
    magnitude = np.sqrt(grad_x[:-1, :] ** 2 + grad_y[:, :-1] ** 2)
    threshold = np.percentile(magnitude, 95) if magnitude.size > 0 else 0.0
    return float(np.sum(magnitude > threshold)) if magnitude.size > 0 else 0.0


def _compute_histogram_balance(gray_arr: np.ndarray) -> float:
    """Compute histogram balance (how evenly distributed pixel values are)."""
    hist, _ = np.histogram(gray_arr, bins=64, range=(0, 256))
    hist_norm = hist.astype(np.float32)
    if hist_norm.sum() > 0:
        hist_norm /= hist_norm.sum()
        balance = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))
    else:
        balance = 0.0
    return balance


def _compute_color_temperature(mean_r: float, mean_g: float, mean_b: float) -> float:
    """Estimate color temperature from RGB means (warm vs cool)."""
    total = mean_r + mean_g + mean_b
    if total > 0:
        r_norm = mean_r / total
        b_norm = mean_b / total
        temp_score = (r_norm - b_norm) * 100.0
    else:
        temp_score = 0.0
    return float(temp_score)


def _compute_center_brightness_ratio(gray_arr: np.ndarray) -> float:
    """Compute ratio of center brightness to overall brightness."""
    h, w = gray_arr.shape
    center_h, center_w = h // 2, w // 2
    center_size = min(h, w) // 4
    center_region = gray_arr[
        center_h - center_size : center_h + center_size, center_w - center_size : center_w + center_size
    ]
    center_brightness = float(np.mean(center_region)) if center_region.size > 0 else 0.0
    overall_brightness = float(np.mean(gray_arr))
    ratio = center_brightness / overall_brightness if overall_brightness > 0 else 1.0
    return float(ratio)


def _compute_exposure_quality(gray_arr: np.ndarray, highlight_clip: float, shadow_clip: float) -> float:
    """Compute exposure quality score (0-1, higher = better exposed)."""
    clipping_penalty = (highlight_clip + shadow_clip) / 200.0
    clipping_score = 1.0 - min(1.0, clipping_penalty)
    hist_range = float(np.max(gray_arr) - np.min(gray_arr)) / 255.0
    quality = (clipping_score + hist_range) / 2.0
    return float(quality)


def _compute_color_diversity(arr: np.ndarray) -> float:
    """Compute color diversity using unique color count approximation."""
    # Ensure arr has valid shape
    if len(arr.shape) < 3 or arr.shape[2] != 3:
        return 0.5  # Default fallback
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return 0.5
    sample_size = min(1000, h * w)
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(h * w, size=sample_size, replace=False)
    sampled = arr.reshape(-1, 3)[indices]
    quantized = (sampled / 32).astype(int)
    unique_colors = len(np.unique(quantized, axis=0))
    diversity = float(unique_colors / sample_size * 1000.0)
    return diversity


def _compute_rule_of_thirds_score(gray_arr: np.ndarray, w: int, h: int) -> float:
    """Compute rule of thirds score (how well subject aligns with thirds lines)."""
    third_w1, third_w2 = w // 3, 2 * w // 3
    third_h1, third_h2 = h // 3, 2 * h // 3
    line_variance_w1 = float(np.var(gray_arr[:, third_w1]))
    line_variance_w2 = float(np.var(gray_arr[:, third_w2]))
    line_variance_h1 = float(np.var(gray_arr[third_h1, :]))
    line_variance_h2 = float(np.var(gray_arr[third_h2, :]))
    thirds_variance = (line_variance_w1 + line_variance_w2 + line_variance_h1 + line_variance_h2) / 4.0
    overall_variance = float(np.var(gray_arr))
    score = thirds_variance / overall_variance if overall_variance > 0 else 0.5
    return float(min(1.0, score))


def _compute_symmetry_score(gray_arr: np.ndarray) -> float:
    """Compute horizontal and vertical symmetry scores."""
    h, w = gray_arr.shape
    left_half = gray_arr[:, : w // 2]
    right_half = gray_arr[:, w // 2 :]
    if right_half.shape[1] > left_half.shape[1]:
        right_half = right_half[:, : left_half.shape[1]]
    elif right_half.shape[1] < left_half.shape[1]:
        left_half = left_half[:, : right_half.shape[1]]
    right_half_flipped = np.fliplr(right_half)
    horizontal_sym = 1.0 - float(np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0)
    top_half = gray_arr[: h // 2, :]
    bottom_half = gray_arr[h // 2 :, :]
    if bottom_half.shape[0] > top_half.shape[0]:
        bottom_half = bottom_half[: top_half.shape[0], :]
    elif bottom_half.shape[0] < top_half.shape[0]:
        top_half = top_half[: bottom_half.shape[0], :]
    bottom_half_flipped = np.flipud(bottom_half)
    vertical_sym = 1.0 - float(np.mean(np.abs(top_half.astype(float) - bottom_half_flipped.astype(float))) / 255.0)
    symmetry = (horizontal_sym + vertical_sym) / 2.0
    return float(symmetry)


def _compute_horizon_levelness(gray_arr: np.ndarray) -> float:
    """Estimate horizon levelness using edge detection."""
    # Skip cv2 in multiprocessing to avoid deadlocks - use simple fallback
    # Simple fallback: measure horizontal edge strength variance
    h, w = gray_arr.shape
    grad_y = np.diff(gray_arr.astype(np.float32), axis=0)
    horizontal_strength = np.sum(np.abs(grad_y), axis=1) / w if w > 0 else np.zeros(h-1)
    if len(horizontal_strength) > 0:
        line_variance = float(np.var(horizontal_strength))
        levelness = 1.0 / (1.0 + line_variance / (h * h)) if h > 0 else 0.5
    else:
        levelness = 0.5
    return float(levelness)


def _compute_center_focus_quality(gray_arr: np.ndarray, overall_sharpness: float) -> float:
    """Compute focus quality in center vs periphery."""
    h, w = gray_arr.shape
    center_h, center_w = h // 2, w // 2
    center_size = min(h, w) // 4
    center_region = gray_arr[
        center_h - center_size : center_h + center_size, center_w - center_size : center_w + center_size
    ]
    if center_region.size > 0:
        p = np.pad(center_region.astype(np.float32), 1, mode="edge")
        laplacian_center = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:] - 4 * p[1:-1, 1:-1]
        center_sharpness = float(laplacian_center.var())
    else:
        center_sharpness = overall_sharpness
    ratio = center_sharpness / overall_sharpness if overall_sharpness > 0 else 1.0
    return float(min(2.0, ratio))


def _compute_dynamic_range_utilization(gray_arr: np.ndarray) -> float:
    """Compute how well the image utilizes the full dynamic range."""
    min_val = float(np.min(gray_arr))
    max_val = float(np.max(gray_arr))
    range_used = (max_val - min_val) / 255.0
    hist, _ = np.histogram(gray_arr, bins=64, range=(0, 256))
    hist_norm = hist.astype(np.float32)
    if hist_norm.sum() > 0:
        hist_norm /= hist_norm.sum()
        bins_used = float(np.sum(hist_norm > 0.01)) / 64.0
    else:
        bins_used = 0.0
    utilization = (range_used + bins_used) / 2.0
    return float(utilization)


def _compute_subject_isolation_score(gray_arr: np.ndarray, arr: np.ndarray) -> float:
    """Detect bokeh/blur gradients - measure depth-of-field separation.
    
    Higher score = better subject isolation (sharp center, blurred edges).
    """
    # Ensure arr matches gray_arr shape
    arr_h, arr_w = arr.shape[:2]
    h, w = gray_arr.shape
    if arr_h != h or arr_w != w:
        # Crop or pad to match exactly
        min_h = min(arr_h, h)
        min_w = min(arr_w, w)
        arr = arr[:min_h, :min_w, :]
        # If arr is smaller, pad with edge values
        if arr_h < h or arr_w < w:
            from numpy import pad
            pad_h = max(0, h - arr_h)
            pad_w = max(0, w - arr_w)
            arr = pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    center_h, center_w = h // 2, w // 2
    center_size = min(h, w) // 3
    
    # Center region sharpness
    center_region = gray_arr[
        max(0, center_h - center_size) : min(h, center_h + center_size),
        max(0, center_w - center_size) : min(w, center_w + center_size)
    ]
    if center_region.size > 0:
        p_center = np.pad(center_region.astype(np.float32), 1, mode="edge")
        laplacian_center = p_center[:-2, :-2] + p_center[:-2, 2:] + p_center[2:, :-2] + p_center[2:, 2:] - 4 * p_center[1:-1, 1:-1]
        center_sharpness = float(laplacian_center.var())
    else:
        center_sharpness = 0.0
    
    # Edge region sharpness (corners and borders)
    edge_mask = np.ones((h, w), dtype=bool)
    edge_mask[
        max(0, center_h - center_size) : min(h, center_h + center_size),
        max(0, center_w - center_size) : min(w, center_w + center_size)
    ] = False
    edge_region = gray_arr[edge_mask]
    
    if edge_region.size > 0:
        # Sample edge region for efficiency
        sample_size = min(1000, edge_region.size)
        np.random.seed(42)
        sampled = np.random.choice(edge_region, size=sample_size, replace=False)
        edge_arr = sampled.reshape(-1, 1) if len(sampled.shape) == 1 else sampled
        p_edge = np.pad(edge_arr.astype(np.float32), (1, 1), mode="edge")
        if p_edge.size > 4:
            laplacian_edge = p_edge[:-2] + p_edge[2:] - 2 * p_edge[1:-1]
            edge_sharpness = float(laplacian_edge.var())
        else:
            edge_sharpness = 0.0
    else:
        edge_sharpness = 0.0
    
    # Isolation score: center sharp / edge blur ratio
    if edge_sharpness > 0:
        isolation = center_sharpness / (edge_sharpness + 1e-6)
    else:
        isolation = 0.0
    
    return float(min(10.0, isolation))  # Cap at 10.0


def _compute_golden_hour_score(hour: float, color_temp: float) -> float:
    """Detect golden hour: warm colors + time of day (sunrise/sunset).
    
    Golden hour: ~6-8 AM or 6-8 PM (hour 6-8 or 18-20) with warm tones.
    """
    # Normalize hour to 0-24
    hour_norm = hour % 24
    
    # Check if in golden hour windows (6-8 AM or 18-20 PM)
    morning_golden = 6.0 <= hour_norm <= 8.0
    evening_golden = 18.0 <= hour_norm <= 20.0
    is_golden_time = morning_golden or evening_golden
    
    # Warm color temperature (positive = warm/red)
    warm_score = max(0.0, color_temp / 50.0)  # Normalize color temp
    
    # Combine time and color
    if is_golden_time:
        golden_score = 0.5 + 0.5 * warm_score
    else:
        golden_score = 0.2 * warm_score  # Some warm color but wrong time
    
    return float(min(1.0, golden_score))


def _compute_lighting_quality_score(gray_arr: np.ndarray, highlight_clip: float, shadow_clip: float) -> float:
    """Assess lighting quality: harsh vs soft (contrast analysis).
    
    Softer, more even lighting is generally preferred.
    """
    # High contrast = harsh lighting
    contrast = float(np.std(gray_arr))
    contrast_score = 1.0 - min(1.0, contrast / 80.0)  # Lower contrast = better
    
    # Clipping penalty (harsh highlights/shadows)
    clipping_penalty = (highlight_clip + shadow_clip) / 200.0
    clipping_score = 1.0 - min(1.0, clipping_penalty)
    
    # Histogram smoothness (even distribution = soft lighting)
    hist, _ = np.histogram(gray_arr, bins=32, range=(0, 256))
    hist_norm = hist.astype(np.float32)
    if hist_norm.sum() > 0:
        hist_norm /= hist_norm.sum()
        # Measure smoothness (low variance in histogram = smoother)
        hist_smoothness = 1.0 - min(1.0, float(np.std(hist_norm)) * 10.0)
    else:
        hist_smoothness = 0.5
    
    # Combine scores
    quality = (contrast_score * 0.4 + clipping_score * 0.3 + hist_smoothness * 0.3)
    return float(min(1.0, quality))


def _compute_color_harmony_score(arr: np.ndarray) -> float:
    """Detect color harmony: complementary/analogous color schemes.
    
    Harmonious colors (complementary or analogous) are more pleasing.
    """
    # Ensure arr has valid shape
    if len(arr.shape) < 3 or arr.shape[2] != 3:
        return 0.5  # Default fallback
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return 0.5
    sample_size = min(500, h * w)
    np.random.seed(42)
    indices = np.random.choice(h * w, size=sample_size, replace=False)
    sampled = arr.reshape(-1, 3)[indices]
    
    # Convert RGB to HSV for color wheel analysis
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Fallback: use RGB ratios
    r_norm = sampled[:, 0] / (sampled.sum(axis=1) + 1e-6)
    g_norm = sampled[:, 1] / (sampled.sum(axis=1) + 1e-6)
    hues = np.arctan2(g_norm - 0.33, r_norm - 0.33) * 180.0 / np.pi
    
    # Measure hue distribution (clustered = harmonious)
    if len(hues) > 0:
        # Circular variance for hues
        hues_rad = np.deg2rad(hues)
        mean_hue = np.arctan2(np.sin(hues_rad).mean(), np.cos(hues_rad).mean())
        variance = np.mean((np.sin(hues_rad - mean_hue) ** 2))
        harmony = 1.0 - min(1.0, variance * 2.0)  # Lower variance = more harmonious
    else:
        harmony = 0.5
    
    return float(harmony)


def _compute_sky_ground_ratio(gray_arr: np.ndarray, h: int) -> float:
    """Detect horizon and measure sky vs ground area ratio.
    
    Landscape composition indicator - rule of thirds for horizon often better.
    """
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Fallback: use brightness gradient
    row_brightness = np.mean(gray_arr, axis=1)
    # Find largest brightness change (likely sky/ground boundary)
    brightness_diff = np.abs(np.diff(row_brightness))
    if len(brightness_diff) > 0:
        horizon_y = int(np.argmax(brightness_diff))
    else:
        horizon_y = h // 2
    
    # Calculate sky/ground ratio
    sky_area = max(1, horizon_y)
    ground_area = max(1, h - horizon_y)
    ratio = sky_area / (sky_area + ground_area)
    
    # Score: rule of thirds (1/3 or 2/3) is better than centered (0.5)
    thirds_score = 1.0 - abs(ratio - 0.33) - abs(ratio - 0.67)  # Higher if near 1/3 or 2/3
    return float(max(0.0, thirds_score))


def _compute_motion_blur_score(gray_arr: np.ndarray) -> float:
    """Detect motion blur: directional blur patterns vs focus blur.
    
    Motion blur has directional patterns, focus blur is uniform.
    """
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Fallback: measure edge directionality using gradients
    grad_x = np.diff(gray_arr.astype(np.float32), axis=1)  # shape: (h, w-1)
    grad_y = np.diff(gray_arr.astype(np.float32), axis=0)  # shape: (h-1, w)
    if grad_x.size > 0 and grad_y.size > 0:
        # Measure directional consistency - match shapes
        angles = np.arctan2(grad_y[:, :-1], grad_x[:-1, :]) * 180.0 / np.pi
        angle_variance = float(np.var(angles))
        motion_score = 1.0 - min(1.0, angle_variance / 3600.0)
    else:
        motion_score = 0.0
    
    return float(motion_score)


def _extract_person_detection(path: str) -> float:
    """Extract person detection feature from object detection cache.
    
    Returns 1.0 if person is detected, 0.0 otherwise.
    Uses module-level cache to avoid repeated disk I/O.
    """
    import time
    t0 = time.perf_counter()
    global _object_detection_cache, _object_detection_cache_mtime
    
    try:
        from src.object_detection import load_object_cache, get_cache_path
        
        # Check if cache needs reloading
        cache_path = get_cache_path()
        cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
        
        if _object_detection_cache is None or cache_mtime != _object_detection_cache_mtime:
            reload_start = time.perf_counter()
            _object_detection_cache = load_object_cache()
            _object_detection_cache_mtime = cache_mtime
            reload_time = time.perf_counter() - reload_start
            logging.debug(f"[features] Reloaded object detection cache in {reload_time*1000:.1f}ms")
        
        cache = _object_detection_cache
        basename = os.path.basename(path)
        detections = cache.get(basename, [])
        
        if not detections:
            elapsed = time.perf_counter() - t0
            if elapsed > 0.01:  # Only log if slow
                logging.debug(f"[features] Person detection (no detections) for {basename}: {elapsed*1000:.1f}ms")
            return 0.0
        
        person_classes = {"person"}
        classes = [d.get("class", "").lower() for d in detections]
        has_person = 1.0 if any(c in person_classes for c in classes) else 0.0
        
        elapsed = time.perf_counter() - t0
        if elapsed > 0.01:  # Only log if slow
            logging.debug(f"[features] Person detection (result={has_person}) for {basename}: {elapsed*1000:.1f}ms")
        
        return has_person
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logging.debug(f"[features] Failed to extract person detection for {path} in {elapsed*1000:.1f}ms: {e}")
        return 0.0


def _do_feature_extraction(prep: ImagePreprocessResult, path: str) -> list[float]:
    import time
    feat_start = time.perf_counter()
    basename = os.path.basename(path)
    timing_breakdown = {}
    
    # Highlight & Shadow Clipping
    clip_start = time.perf_counter()
    highlight_clip = float(np.sum(prep.gray_arr >= 250) / prep.gray_arr.size * 100)
    shadow_clip = float(np.sum(prep.gray_arr <= 5) / prep.gray_arr.size * 100)
    clip_time = time.perf_counter() - clip_start

    # Sharpness: Fixed - use padding instead of roll to avoid boundary wraparound artifacts
    h, w = prep.gray_arr.shape
    p = np.pad(prep.gray_arr, 1, mode="edge").astype(np.float32)
    # Ensure all slices have same shape (h, w)
    laplacian = p[0:h, 0:w] + p[0:h, 2:w+2] + p[2:h+2, 0:w] + p[2:h+2, 2:w+2] - 4 * p[1:h+1, 1:w+1]
    sharpness = float(laplacian.var())

    # Noise Level: high-frequency component via gaussian blur
    # Skip cv2 in multiprocessing to avoid deadlocks - use fallback always
    # Use simple variance-based noise estimate to avoid shape issues
    noise_start = time.perf_counter()
    # Simple approach: measure high-frequency variance using Laplacian
    p = np.pad(prep.gray_arr, 1, mode="edge").astype(np.float32)
    h, w = prep.gray_arr.shape
    laplacian = p[1:h+1, 1:w+1] * 4 - (p[0:h, 1:w+1] + p[2:h+2, 1:w+1] + p[1:h+1, 0:w] + p[1:h+1, 2:w+2])
    noise_level = float(laplacian.var())
    timing_breakdown['noise'] = (time.perf_counter() - noise_start) * 1000

    # Saturation: Use already-downsampled image (no additional resize needed)
    # Image is already downsampled to 1MP max in preprocessing, which is fine for saturation
    sat_start = time.perf_counter()
    arr = np.array(prep.img)
    
    # Optimized: use faster numpy operations for min/max
    # Reshape to (h*w, 3) for faster min/max computation
    arr_flat = arr.reshape(-1, 3).astype(np.float32)
    max_val = np.maximum.reduce(arr_flat, axis=1)
    min_val = np.minimum.reduce(arr_flat, axis=1)
    # HSV saturation: (max-min)/max when max > 0
    mask = max_val > 0
    sat_values = np.zeros_like(max_val)
    sat_values[mask] = (max_val[mask] - min_val[mask]) / max_val[mask]
    saturation = float(np.mean(sat_values) * 100)
    timing_breakdown['saturation'] = (time.perf_counter() - sat_start) * 1000

    # Entropy - optimized: use bincount for faster histogram
    entropy_start = time.perf_counter()
    gray_uint8 = prep.gray_arr.astype(np.uint8)
    bin_width = 256.0 / 64
    bin_indices = (gray_uint8.flatten() / bin_width).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, 63)
    gray_hist = np.bincount(bin_indices, minlength=64).astype(np.float32)
    if gray_hist.sum():
        p = gray_hist / gray_hist.sum()
        entropy = float(-(p * np.log2(p + 1e-12)).sum())
    else:
        entropy = 0.0
    timing_breakdown['entropy'] = (time.perf_counter() - entropy_start) * 1000

    # Temporal features: Extract from EXIF datetime first, fallback to file mtime
    hour = 12.0
    day_of_week = 3.0
    month = 6.0
    is_weekend = 0.0
    try:
        from datetime import datetime

        dt = None

        # Try EXIF datetime first (from exif_data parameter)
        if prep.exif_data.get("datetime_str"):
            try:
                dt = datetime.strptime(prep.exif_data["datetime_str"], "%Y:%m:%d %H:%M:%S")
            except Exception:
                logging.exception("Error serializing feature block")
                raise

        # Fallback to file modification time
        if dt is None:
            mtime = os.path.getmtime(path)
            dt = datetime.fromtimestamp(mtime)

        hour = float(dt.hour)
        day_of_week = float(dt.weekday())
        month = float(dt.month)
        is_weekend = float(1 if dt.weekday() >= 5 else 0)
    except Exception:
        logging.exception("Error building features")
        raise

    # Use EXIF data extracted before RGB conversion (passed as parameter)
    iso = float(np.log1p(prep.exif_data["iso"]))
    aperture = float(prep.exif_data["aperture"])
    shutter_speed = float(np.log1p(prep.exif_data["shutter_speed"]))
    flash_fired = float(prep.exif_data["flash_fired"])
    focal_length_35mm = float(prep.exif_data["focal_length_35mm"])
    digital_zoom = float(prep.exif_data["digital_zoom"])
    exposure_compensation = float(prep.exif_data["exposure_compensation"])
    white_balance = float(prep.exif_data["white_balance"])
    exposure_mode = float(prep.exif_data["exposure_mode"])
    metering_mode = float(prep.exif_data["metering_mode"])

    # Advanced features: implement new feature candidates
    advanced_start = time.perf_counter()
    feature_times = {}
    # Removed try-except wrapper - let exceptions propagate to catch bugs
    t0 = time.perf_counter()
    edge_density, edge_strength = _compute_edge_features(prep.gray_arr)
    feature_times['edge'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    corner_count = _compute_corner_count(prep.gray_arr)
    feature_times['corner'] = (time.perf_counter() - t0) * 1000
    
    face_count = 0.0  # Would require face detection model
    
    t0 = time.perf_counter()
    histogram_balance = _compute_histogram_balance(prep.gray_arr)
    feature_times['hist_balance'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    color_temperature = _compute_color_temperature(prep.mean_r, prep.mean_g, prep.mean_b)
    feature_times['color_temp'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    center_brightness_ratio = _compute_center_brightness_ratio(prep.gray_arr)
    feature_times['center_bright'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    exposure_quality = _compute_exposure_quality(prep.gray_arr, highlight_clip, shadow_clip)
    feature_times['exposure'] = (time.perf_counter() - t0) * 1000
    
    # Ensure arr shape matches gray_arr for color features (critical - re-validate before use)
    # Re-extract arr from prep.img to ensure correct shape, as arr may have been modified
    t0 = time.perf_counter()
    arr = np.array(prep.img)
    arr_h, arr_w = arr.shape[:2] if len(arr.shape) >= 2 else (0, 0)
    gray_h, gray_w = prep.gray_arr.shape
    # Handle transpose
    if arr_h != gray_h or arr_w != gray_w:
        if arr_h == gray_w and arr_w == gray_h:
            arr = arr.transpose(1, 0, 2)
            arr_h, arr_w = arr.shape[:2]
    # Final crop/pad to match exactly
    if arr_h != gray_h or arr_w != gray_w or len(arr.shape) < 3 or arr.shape[2] != 3:
        if len(arr.shape) < 3 or arr.shape[2] != 3:
            # Invalid shape - return default
            arr = np.zeros((gray_h, gray_w, 3), dtype=np.uint8)
        else:
            # Crop or pad to match gray_arr dimensions
            min_h = min(arr_h, gray_h)
            min_w = min(arr_w, gray_w)
            arr = arr[:min_h, :min_w, :]
            if arr_h < gray_h or arr_w < gray_w:
                # Pad with edge values if arr is smaller
                pad_h = max(0, gray_h - arr_h)
                pad_w = max(0, gray_w - arr_w)
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    feature_times['arr_prep'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    color_diversity = _compute_color_diversity(arr)
    feature_times['color_diversity'] = (time.perf_counter() - t0) * 1000
    
    # Process all images the same way - no skipping expensive features
    t0 = time.perf_counter()
    rule_of_thirds_score = _compute_rule_of_thirds_score(prep.gray_arr, prep.w, prep.h)
    feature_times['rule_thirds'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    symmetry_score = _compute_symmetry_score(prep.gray_arr)
    feature_times['symmetry'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    horizon_levelness = _compute_horizon_levelness(prep.gray_arr)
    feature_times['horizon'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    center_focus_quality = _compute_center_focus_quality(prep.gray_arr, sharpness)
    feature_times['center_focus'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    dynamic_range_utilization = _compute_dynamic_range_utilization(prep.gray_arr)
    feature_times['dynamic_range'] = (time.perf_counter() - t0) * 1000
    
    # New photography-specific features
    t0 = time.perf_counter()
    subject_isolation = _compute_subject_isolation_score(prep.gray_arr, arr)
    feature_times['subject_isolation'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    golden_hour = _compute_golden_hour_score(hour, color_temperature)
    feature_times['golden_hour'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    lighting_quality = _compute_lighting_quality_score(prep.gray_arr, highlight_clip, shadow_clip)
    feature_times['lighting'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    color_harmony = _compute_color_harmony_score(arr)
    feature_times['color_harmony'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    sky_ground_ratio = _compute_sky_ground_ratio(prep.gray_arr, prep.h)
    feature_times['sky_ground'] = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    motion_blur = _compute_motion_blur_score(prep.gray_arr)
    feature_times['motion_blur'] = (time.perf_counter() - t0) * 1000
    
    advanced_time = time.perf_counter() - advanced_start
    if advanced_time > 0.5 or any(t > 50 for t in feature_times.values()):
        # Log slow features sorted by time
        slow_features = sorted(feature_times.items(), key=lambda x: x[1], reverse=True)
        slow_list = ", ".join([f"{name}={t:.1f}ms" for name, t in slow_features[:8] if t > 5])
        if slow_list:
            logging.info(f"[features] Feature timings for {basename}: total={advanced_time*1000:.1f}ms ({slow_list})")

    # Return 71 core features (FAST mode)
    features: list[float] = [
        float(prep.w),
        float(prep.h),
        float(prep.aspect),
        float(np.log1p(prep.file_size)),
        float(prep.mean_brightness),
        float(prep.std_brightness),
        float(prep.mean_r),
        float(prep.mean_g),
        float(prep.mean_b),
        float(prep.std_r),
        float(prep.std_g),
        float(prep.std_b),
    ]
    features.extend(prep.hist_feat.tolist())
    features.extend([
        float(sharpness),
        float(saturation),
        float(entropy),
        float(prep.std_brightness),
        float(highlight_clip),
        float(shadow_clip),
        float(noise_level),
        float(hour),
        float(day_of_week),
        float(month),
        float(is_weekend),
        float(iso),
        float(aperture),
        float(shutter_speed),
        float(flash_fired),
        float(focal_length_35mm),
        float(digital_zoom),
        float(exposure_compensation),
        float(white_balance),
        float(exposure_mode),
        float(metering_mode),
        float(edge_density),
        float(edge_strength),
        float(corner_count),
        float(face_count),
        float(histogram_balance),
        float(color_temperature),
        float(center_brightness_ratio),
        float(exposure_quality),
        float(color_diversity),
        float(rule_of_thirds_score),
        float(symmetry_score),
        float(horizon_levelness),
        float(center_focus_quality),
        float(dynamic_range_utilization),
        float(subject_isolation),
        float(golden_hour),
        float(lighting_quality),
        float(color_harmony),
        float(sky_ground_ratio),
        float(motion_blur),
    ])
    
    # Person detection from object detection cache (added as new feature)
    person_start = time.perf_counter()
    person_det = float(_extract_person_detection(path))
    person_time = time.perf_counter() - person_start
    if person_time > 0.1:
        logging.info(f"[features] SLOW person detection for {basename}: {person_time*1000:.1f}ms")
    features.append(float(person_det))
    
    feat_total = time.perf_counter() - feat_start
    if feat_total > 1.5:
        # Log breakdown of slow operations
        slow_ops = sorted(timing_breakdown.items(), key=lambda x: x[1], reverse=True)
        slow_list = ", ".join([f"{name}={t:.1f}ms" for name, t in slow_ops[:8] if t > 20])
        logging.info(f"[features] SLOW feature extraction for {basename}: total={feat_total:.2f}s (clip={clip_time*1000:.1f}ms, advanced={advanced_time:.2f}s, person={person_time*1000:.1f}ms, {slow_list})")
    
    return features  # type: ignore[return-value]


def extract_features(path: str) -> list[float] | None:
    import time
    t0 = time.perf_counter()
    global feature_cache
    if feature_cache is None:
        feature_cache = load_feature_cache()
    ap = os.path.abspath(path)
    if feature_cache is not None and ap in feature_cache:
        cached = feature_cache[ap]
        if isinstance(cached, (list, tuple)) and len(cached) == FEATURE_COUNT:
            t1 = time.perf_counter()
            return list(cached)  # type: ignore[return-value]
        else:
            try:
                del feature_cache[ap]
                save_feature_cache(feature_cache)
            except Exception:
                logging.exception("[features] Failed to delete stale cache entry for %s", ap)
    prep = _preprocess_image(path)
    if prep is None:
        logging.warning(f"[features] Preprocessing failed for {path}")
        return None
    try:
        feats = _do_feature_extraction(prep, path)
        t1 = time.perf_counter()
        if len(feats) != FEATURE_COUNT:
            logging.warning(f"[features] Length mismatch {len(feats)} expected {FEATURE_COUNT} path={path}")
        if feature_cache is not None:
            feature_cache[ap] = feats
            # Save cache immediately after each extraction to prevent data loss
            try:
                save_feature_cache(feature_cache)
            except Exception as e:
                logging.warning(f"[features] Failed to save cache after extraction for {path}: {e}")
        logging.debug(f"[features] Extracted {len(feats)} features for {os.path.basename(path)}")
        return feats
    except Exception as e:
        logging.warning(f"[features] Extraction failed for {path}: {e}")
        return None


def _extract_single_feature(path: str):
    import time
    t0 = time.perf_counter()
    ap = os.path.abspath(path)
    basename = os.path.basename(path)
    
    prep_start = time.perf_counter()
    prep = _preprocess_image(path)
    prep_time = time.perf_counter() - prep_start
    if prep is None:
        logging.debug(f"[features] Preprocessing failed for {basename} in {prep_time*1000:.1f}ms")
        return (ap, None)
    
    try:
        extract_start = time.perf_counter()
        feats = _do_feature_extraction(prep, path)
        extract_time = time.perf_counter() - extract_start
        total_time = time.perf_counter() - t0
        # Log slow extractions (>1s) at INFO level, others at DEBUG
        if total_time > 1.0:
            logging.info(f"[features] SLOW extraction for {basename}: prep={prep_time*1000:.1f}ms, extract={extract_time*1000:.1f}ms, total={total_time:.2f}s")
        else:
            logging.debug(f"[features] Extracted {len(feats)} features for {basename}: prep={prep_time*1000:.1f}ms, extract={extract_time*1000:.1f}ms, total={total_time*1000:.1f}ms")
        return (ap, feats)
    except Exception as e:
        total_time = time.perf_counter() - t0
        logging.warning(f"[features] Single extraction failed for {basename} in {total_time*1000:.1f}ms: {e}")
        return (ap, None)


def _check_cache_hits(paths: list[str], cache: dict, progress_callback) -> tuple[list, list, list, int]:
    """Check which paths are cached and which need extraction."""
    results: list[list[float] | None] = [None] * len(paths)
    needed = []
    needed_idx = []
    hits = 0

    for i, p in enumerate(paths):
        ap = os.path.abspath(p)
        if ap in cache:
            cached = cache[ap]
            if isinstance(cached, (list, tuple)) and len(cached) == FEATURE_COUNT:
                results[i] = list(cached)  # type: ignore[assignment]
                hits += 1
                if progress_callback:
                    # Update progress for cached items, but throttle to avoid UI spam
                    # Update every 10 items or at milestones
                    if hits % 10 == 0 or hits == 1 or hits == len(paths) or i == len(paths) - 1:
                        progress_callback(hits, len(paths), f"cached {hits}/{len(paths)}")
                continue
            else:
                cache.pop(ap, None)
        needed.append(p)
        needed_idx.append(i)

    return results, needed, needed_idx, hits


def _parallel_extract(paths: list[str], progress_callback, hits: int, total: int):
    """Extract features in parallel using multiprocessing."""
    import os
    import time
    import atexit
    import threading
    from multiprocessing import Pool, cpu_count, get_start_method, set_start_method
    
    # Disable OpenCV threading to avoid deadlocks in multiprocessing
    # This must be done before importing cv2 in worker processes
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    
    # Ensure proper start method on macOS (spawn is default, but be explicit)
    try:
        current_method = get_start_method(allow_none=True)
        if current_method is None:
            set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    parallel_start = time.perf_counter()
    
    # Use more workers for better parallelism (cap at 4 to limit resource usage)
    max_workers = int(os.environ.get("FEATURE_EXTRACT_WORKERS", min(cpu_count(), 4)))
    workers = min(max_workers, len(paths))
    # In pytest environments force a single worker to avoid multiprocessing deadlocks
    if os.environ.get("PYTEST_CURRENT_TEST"):
        workers = 1

    if progress_callback:
        progress_callback(hits, total, f"workers={workers}")

    logging.info(f"[features] Starting parallel extraction: {len(paths)} images with {workers} workers")
    extracted = []
    pool_start = time.perf_counter()
    pool = None
    global _active_pool, _pool_lock
    
    # Initialize lock if needed
    if _pool_lock is None:
        _pool_lock = threading.Lock()
    
    try:
        # Use lock to ensure only one pool exists at a time (max 4 workers total)
        with _pool_lock:
            # Wait for any existing pool to finish before creating a new one
            if _active_pool is not None:
                try:
                    logging.info(f"[features] Waiting for existing pool to finish (PID={os.getpid()})")
                    # Close the pool to prevent new tasks, then wait for completion
                    try:
                        _active_pool.close()
                    except ValueError:
                        # Pool already closed
                        pass
                    # Wait for pool to finish (no timeout parameter in Python 3.12)
                    _active_pool.join()
                    # If still not done, terminate
                    if hasattr(_active_pool, '_processes') and _active_pool._processes:
                        alive = any(p.is_alive() for p in _active_pool._processes if p is not None)
                        if alive:
                            logging.warning(f"[features] Existing pool still active, terminating (PID={os.getpid()})")
                            _active_pool.terminate()
                            _active_pool.join()
                            # Force kill remaining processes
                            for p in _active_pool._processes:
                                if p is not None and p.is_alive():
                                    p.terminate()
                                    p.join()
                                    if p.is_alive():
                                        p.kill()
                except Exception as e:
                    logging.warning(f"[features] Error waiting for existing pool: {e}")
                _active_pool = None
            
            # Create new pool (ensures max 4 workers total across all extractions)
            pool = Pool(processes=workers)
            _active_pool = pool
            logging.info(f"[features] Created multiprocessing pool with {workers} workers (PID={os.getpid()})")
            
            # Keep lock held during extraction to prevent concurrent pools
            for idx, r in enumerate(pool.imap(_extract_single_feature, paths)):
                extracted.append(r)
                current = hits + idx + 1
                if progress_callback:
                    # Update progress for every image to ensure UI stays responsive
                    progress_callback(current, total, f"extracting {idx + 1}/{len(paths)}")
                # Log every 10% or at key milestones
                if (idx + 1) % max(1, len(paths) // 10) == 0 or idx == 0 or idx == len(paths) - 1:
                    elapsed = time.perf_counter() - pool_start
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    logging.info(f"[features] Parallel progress: {idx + 1}/{len(paths)} ({rate:.1f} img/s)")
            
            # Clear pool reference before releasing lock
            _active_pool = None
    finally:
        # Close pool but keep reference for cleanup tracking
        if pool is not None:
            pool.close()
            # Don't join here - let next extraction handle cleanup via lock
            # This allows the lock to serialize pool creation/destruction

    parallel_time = time.perf_counter() - parallel_start
    rate = len(extracted) / parallel_time if parallel_time > 0 else 0
    logging.info(f"[features] Parallel extraction complete: {len(extracted)} images in {parallel_time:.2f}s ({rate:.1f} img/s)")
    return extracted


def _sequential_extract(paths: list[str], progress_callback, hits: int, total: int):
    """Extract features sequentially."""
    import time
    seq_start = time.perf_counter()
    logging.info(f"[features] Starting sequential extraction: {len(paths)} images")
    extracted = []
    for idx, p in enumerate(paths):
        r = _extract_single_feature(p)
        extracted.append(r)
        current = hits + idx + 1
        if progress_callback:
            # Update progress for every image to ensure UI stays responsive
            progress_callback(current, total, f"extracting {idx + 1}/{len(paths)}")
        # Log every 10% or at key milestones
        if (idx + 1) % max(1, len(paths) // 10) == 0 or idx == 0 or idx == len(paths) - 1:
            elapsed = time.perf_counter() - seq_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logging.info(f"[features] Sequential progress: {idx + 1}/{len(paths)} ({rate:.1f} img/s)")

    seq_time = time.perf_counter() - seq_start
    rate = len(extracted) / seq_time if seq_time > 0 else 0
    logging.info(f"[features] Sequential extraction complete: {len(extracted)} images in {seq_time:.2f}s ({rate:.1f} img/s)")
    return extracted


def batch_extract_features(paths: list[str], progress_callback=None) -> list[list[float] | None]:
    import time
    batch_start = time.perf_counter()
    global feature_cache, _object_detection_cache, _object_detection_cache_mtime
    
    logging.info(f"[features] ===== BATCH EXTRACTION START: {len(paths)} images =====")
    
    # Load feature cache
    cache_load_start = time.perf_counter()
    if feature_cache is None:
        feature_cache = load_feature_cache()
    cache_load_time = time.perf_counter() - cache_load_start
    logging.info(f"[features] Feature cache loaded in {cache_load_time*1000:.1f}ms: {len(feature_cache) if feature_cache else 0} entries")

    # Pre-load object detection cache once for the entire batch (lazy load in workers)
    # Don't load here - let workers load it on first use to avoid blocking main thread
    # The cache is already module-level cached, so each worker will load it once

    total = len(paths)
    if progress_callback:
        progress_callback(0, total, f"mode={'FULL' if USE_FULL_FEATURES else 'FAST'} features={FEATURE_COUNT}")

    # Check cache hits
    cache_check_start = time.perf_counter()
    results, needed, needed_idx, hits = _check_cache_hits(paths, feature_cache, progress_callback)
    cache_check_time = time.perf_counter() - cache_check_start
    logging.info(f"[features] Cache check completed in {cache_check_time*1000:.1f}ms: {hits}/{total} hits ({100*hits/total:.1f}%)")

    if progress_callback:
        progress_callback(hits, total, f"cache hits={hits} extracting={len(needed)}")

    logging.info(f"[features] Need to extract: {len(needed)} images")

    if needed:
        # In test mode (pytest) avoid spawning multiprocessing pools or heavy parallel work
        extraction_start = time.perf_counter()
        if os.environ.get("PYTEST_CURRENT_TEST"):
            logging.info(f"[features] PYTEST detected: forcing sequential extraction ({len(needed)} images)")
            extracted = _sequential_extract(needed, progress_callback, hits, total)
        else:
            if len(needed) > 10:
                logging.info(f"[features] Using parallel extraction ({len(needed)} images)")
                extracted = _parallel_extract(needed, progress_callback, hits, total)
            else:
                logging.info(f"[features] Using sequential extraction ({len(needed)} images)")
                extracted = _sequential_extract(needed, progress_callback, hits, total)
        extraction_time = time.perf_counter() - extraction_start
        logging.info(f"[features] Extraction phase completed in {extraction_time:.2f}s ({extraction_time/len(needed)*1000:.1f}ms per image)")

        # Process results and save cache incrementally
        save_start = time.perf_counter()
        extracted_count = 0
        failed_count = 0
        for (apath, feats), orig_idx in zip(extracted, needed_idx):
            if feats:
                results[orig_idx] = feats
                feature_cache[apath] = feats
                extracted_count += 1
                # Save cache immediately after each extraction to prevent data loss
                try:
                    save_feature_cache(feature_cache)
                except Exception as e:
                    logging.warning(f"[features] Failed to save cache after extraction for {apath}: {e}")
            else:
                failed_count += 1

        save_time = time.perf_counter() - save_start
        logging.info(
            f"[features] Results processed in {save_time*1000:.1f}ms: {extracted_count} extracted, {failed_count} failed | Cache saved incrementally with {len(feature_cache)} entries"
        )
    else:
        logging.info(f"[features] All {total} images already cached")

    batch_time = time.perf_counter() - batch_start
    logging.info(f"[features] ===== BATCH EXTRACTION COMPLETE: {batch_time:.2f}s total ({batch_time/len(paths)*1000:.1f}ms per image) =====")
    return results


def safe_initialize_feature_cache(preserve_empty: bool = True) -> dict:
    global feature_cache
    if feature_cache is not None:
        return feature_cache

    if not os.path.exists(FEATURE_CACHE_PATH):
        feature_cache = {}
        logging.info(f"[features] init: no cache {FEATURE_CACHE_PATH}")
        return feature_cache

    try:
        with open(FEATURE_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

        cache, key_changed = _migrate_cache_keys(cache)
        cache, removed = _validate_cache_entries(cache)

        if key_changed:
            logging.info("[features] init: migrated keys")
        if removed:
            logging.info(f"[features] init: purged {removed} stale entries")

        if not cache and not preserve_empty:
            os.remove(FEATURE_CACHE_PATH)
            logging.info("[features] init: removed empty cache file")

        feature_cache = cache
        logging.info(f"[features] init: loaded {len(cache)} valid entries (FEATURE_COUNT={FEATURE_COUNT})")
        return feature_cache
    except Exception as e:
        logging.warning(f"[features] init: failed loading cache: {e}")
        feature_cache = {}
        return feature_cache


__all__ = [
    "FEATURE_COUNT",
    "USE_FULL_FEATURES",
    "extract_features",
    "batch_extract_features",
    "load_feature_cache",
    "save_feature_cache",
    "safe_initialize_feature_cache",
]
