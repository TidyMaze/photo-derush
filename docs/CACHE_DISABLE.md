# Cache Disable Feature

## Overview

All caches can be disabled via the `PHOTO_DERUSH_DISABLE_CACHE` environment variable. When set to a truthy value (`1`, `true`, `yes`, `on`), all cache lookups return MISS (None/empty) to force recomputation.

## Environment Variable

```bash
export PHOTO_DERUSH_DISABLE_CACHE=1
```

## All Caches

### Disk Caches

1. **ThumbnailCache** (`src/cache.py`)
   - Location: `~/.photo-derush-cache/*.png`
   - Purpose: Caches generated thumbnails
   - Methods: `get_thumbnail()`, `set_thumbnail()`

2. **PerceptualHashCache** (`src/phash_cache.py`)
   - Location: `~/.photo-derush-cache/perceptual_hashes.json`
   - Purpose: Caches perceptual hashes for duplicate detection
   - Methods: `get_hash()`, `set_hash()`

3. **Feature Cache** (`src/features.py`)
   - Location: `feature_cache.pkl` (project root)
   - Purpose: Caches extracted image features
   - Methods: `load_feature_cache()`, `save_feature_cache()`, `extract_features()`

4. **Object Detection Cache** (`src/object_detection.py`)
   - Location: `.cache/object_detections.joblib`
   - Purpose: Caches object detection results
   - Methods: `load_object_cache()`, `save_object_cache()`, `get_objects_for_image()`, `get_objects_for_images()`

5. **Duplicate Groups Cache** (`src/duplicate_grouping.py`)
   - Location: `~/.photo-derush-cache/duplicate_groups_*.pkl`
   - Purpose: Caches duplicate group assignments
   - Methods: `_load_cached_groups()`, `_save_cached_groups()`, `create_duplicate_groups()`

### In-Memory Caches

6. **Stat Cache** (`src/cache.py`)
   - Purpose: Caches `os.stat()` results for 1 second
   - Variables: `_stat_cache`, `_stat_cache_time`

7. **LazyImageLoader LRU Cache** (`src/lazy_loader.py`)
   - Purpose: LRU cache for EXIF and thumbnail loading
   - Methods: `_cached_exif`, `_cached_thumb`

8. **Pixmap Cache** (`src/view.py`)
   - Purpose: In-memory cache for QPixmaps
   - Variables: `_pixmap_cache`, `_converted_pixmap_cache`
   - Methods: `_pixmap_cache_get()`, `_pixmap_cache_set()`

9. **Overlay Cache** (`src/overlay_widget.py`)
   - Purpose: In-memory cache for overlay pixmaps
   - Variable: `_overlay_cache`

10. **Model/Weights Cache** (`src/object_detection.py`)
    - Purpose: Caches loaded detection models per (backend, device)
    - Variables: `model_cache`, `weights_cache`

11. **Module-Level Object Cache Cache** (`src/object_detection.py`)
    - Purpose: Module-level cache for object detection cache file
    - Variables: `_object_cache_cache`, `_object_cache_mtime`

## Implementation

All cache lookups check `is_cache_disabled()` from `src/cache_config.py` before:
- Reading from cache (returns None/empty if disabled)
- Writing to cache (skips write if disabled)

This ensures that when disabled, all caches behave as if they always MISS, forcing fresh computation.

