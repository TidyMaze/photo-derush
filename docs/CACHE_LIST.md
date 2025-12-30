# All Caches - Status Report

## Environment Variable
`PHOTO_DERUSH_DISABLE_CACHE` - Set to `1`, `true`, `yes`, or `on` to disable all covered caches.

---

## Disk Caches

### ✅ 1. ThumbnailCache
- **File**: `src/cache.py`
- **Location**: `~/.photo-derush-cache/*.png`
- **Purpose**: Caches generated thumbnails to avoid re-generating them
- **Status**: ✅ **COVERED** - `get_thumbnail()` returns None, `set_thumbnail()` skips write

### ✅ 2. PerceptualHashCache
- **File**: `src/phash_cache.py`
- **Location**: `~/.photo-derush-cache/perceptual_hashes.json`
- **Purpose**: Caches perceptual hashes for duplicate detection
- **Status**: ✅ **COVERED** - `get_hash()` returns None, `set_hash()` and `save()` skip write

### ✅ 3. Feature Cache
- **File**: `src/features.py`
- **Location**: `feature_cache.pkl` (project root)
- **Purpose**: Caches extracted image features (78-102 floats per image)
- **Status**: ✅ **COVERED** - `load_feature_cache()` returns {}, `save_feature_cache()` skips write, `extract_features()` bypasses cache lookup

### ✅ 4. Object Detection Cache
- **File**: `src/object_detection.py`
- **Location**: `.cache/object_detections.joblib`
- **Purpose**: Caches object detection results per image
- **Status**: ✅ **COVERED** - `load_object_cache()` returns {}, `save_object_cache()` skips write, cache lookups bypassed in `get_objects_for_image()` and `get_objects_for_images()`

### ✅ 5. Duplicate Groups Cache
- **File**: `src/duplicate_grouping.py`
- **Location**: `~/.photo-derush-cache/duplicate_groups_*.pkl`
- **Purpose**: Caches duplicate group assignments for cross-validation
- **Status**: ✅ **COVERED** - `_load_cached_groups()` returns None, `_save_cached_groups()` skips write

**Detailed Explanation:**

This cache stores **perceptual hash-based grouping** of images to prevent data leakage in machine learning cross-validation.

**Why it exists:**
- When training a model, you need to split data into train/test sets
- If similar images (bursts, near-duplicates, same session) end up in both sets, the model "cheats" by seeing similar data in training
- This inflates accuracy metrics and doesn't reflect real-world performance

**How it works:**
1. Computes perceptual hash (phash) for each image using `imagehash.phash()`
2. Groups images with Hamming distance ≤ 5 (configurable threshold)
3. Returns a list of group IDs (one per filename)
4. Uses `StratifiedGroupKFold` cross-validation to ensure entire groups stay together (all in train OR all in test, never split)

**What's cached:**
- Group assignments: `[group_id_0, group_id_1, ...]` (one per filename)
- Filenames list (for validation)
- File modification times (mtimes) for cache invalidation

**Cache validation:**
- Checks exact filename match (order matters)
- Checks all file mtimes match (detects file changes)
- If either fails, cache is invalidated and groups are recomputed

**Performance:**
- Computing perceptual hashes for thousands of images can take seconds
- Cache allows instant reuse when filenames/mtimes haven't changed
- Cache key includes directory hash, so different directories have separate caches

---

## In-Memory Caches

### ✅ 6. Stat Cache
- **File**: `src/cache.py`
- **Variables**: `_stat_cache`, `_stat_cache_time`
- **Purpose**: Caches `os.stat()` results for 1 second to reduce I/O
- **Status**: ✅ **COVERED** - `_thumbnail_cache_key()` always computes fresh stat when disabled

### ✅ 7. LazyImageLoader LRU Cache
- **File**: `src/lazy_loader.py`
- **Variables**: `_cached_exif`, `_cached_thumb` (LRU-wrapped functions)
- **Purpose**: LRU cache for EXIF and thumbnail loading (max 256 entries each)
- **Status**: ✅ **COVERED** - Calls bypass LRU cache and use `_load_exif_uncached()` / `_load_thumb_uncached()` directly

### ✅ 8. Pixmap Cache
- **File**: `src/view.py`
- **Variables**: `_pixmap_cache`, `_converted_pixmap_cache`
- **Purpose**: In-memory cache for QPixmaps (max 256/512 entries)
- **Status**: ✅ **COVERED** - `_pixmap_cache_get()` returns None, `_pixmap_cache_set()` skips write

### ✅ 9. Overlay Cache
- **File**: `src/overlay_widget.py`
- **Variable**: `_overlay_cache`
- **Purpose**: In-memory cache for overlay pixmaps (max 512 entries)
- **Status**: ✅ **COVERED** - Cache lookup bypassed, cache write skipped in `paintEvent()`

### ❌ 10. Model/Weights Cache
- **File**: `src/object_detection.py`
- **Variables**: `_detection_ctx.model_cache`, `_detection_ctx.weights_cache`
- **Purpose**: Caches loaded detection models per (backend, device) to avoid reloading
- **Status**: ❌ **NOT COVERED** - Intentionally excluded (reloading models on every call would be extremely expensive)

### ✅ 11. Module-Level Object Cache Cache
- **File**: `src/object_detection.py`
- **Variables**: `_object_cache_cache`, `_object_cache_mtime`
- **Purpose**: Module-level cache for object detection cache file to avoid repeated disk loads
- **Status**: ✅ **COVERED** - Cache loading bypassed when disabled

---

## Summary

- **Total Caches**: 11
- **Covered by Disable**: 10 ✅
- **Not Covered**: 1 ❌ (Model cache - intentionally excluded for performance)

All covered caches return MISS (None/empty) when `PHOTO_DERUSH_DISABLE_CACHE` is enabled, forcing fresh computation.

