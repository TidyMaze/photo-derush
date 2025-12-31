# Optimization: load_exif - Pre-load EXIF in Background Thread

## Current Problem

**Hotspot**: `load_exif` - 13.740s cumulative (593 calls)
- **Location**: `src/model.py:67` - `ImageModel.load_exif()`
- **Issue**: EXIF is loaded on-demand, blocking the main thread
- **First call per image**: ~23ms (slow file I/O)
- **Cached calls**: Fast (~0.1ms), but first access is slow

## Current Implementation

```python
# src/model.py:67-98
def load_exif(self, path):
    # Check cache first
    if path in self._exif_cache:
        return self._exif_cache[path]
    
    # Load EXIF (blocking I/O)
    with Image.open(path) as img:
        exif_data = img._getexif()
        # ... process EXIF ...
        result = exif
    self._exif_cache[path] = result
    return result
```

**Problem**: 
- First access to each image's EXIF blocks for ~23ms
- 593 images × 23ms = ~13.6s total blocking time
- Blocks UI thread during initial load

## Solution: Pre-load EXIF in Background Thread

### Strategy

1. **Background pre-loading**: Load EXIF for all images in background thread
2. **Non-blocking access**: Return cached data immediately, or None if not ready
3. **Progressive loading**: Prioritize visible images first
4. **Lazy fallback**: If EXIF not ready, load synchronously (rare case)

### Architecture

```
┌─────────────────┐
│  Main Thread    │
│  (UI)           │
└────────┬────────┘
         │
         │ 1. Request EXIF
         │ 2. Return cached (if ready)
         │ 3. Return None (if not ready)
         │
         ▼
┌─────────────────┐
│  EXIF Cache     │
│  (Thread-safe)  │
└────────┬────────┘
         │
         │ 4. Check cache
         │ 5. Trigger load if missing
         │
         ▼
┌─────────────────┐
│ Background      │
│ Thread Pool     │
│ (4 workers)     │
└─────────────────┘
```

## Implementation Plan

### Step 1: Create Thread-Safe EXIF Cache

```python
# src/model.py

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

class ImageModel:
    def __init__(self, ...):
        # ... existing code ...
        
        # Thread-safe EXIF cache
        self._exif_cache: dict[str, dict] = {}
        self._exif_cache_lock = threading.RLock()
        
        # Track loading status
        self._exif_loading: dict[str, Future] = {}
        self._exif_loading_lock = threading.RLock()
        
        # Background thread pool for EXIF loading
        self._exif_executor = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="exif-loader"
        )
        
        # Track which images need EXIF loaded
        self._exif_pending: set[str] = set()
        self._exif_pending_lock = threading.Lock()
```

### Step 2: Non-Blocking load_exif

```python
def load_exif(self, path: str, blocking: bool = False) -> Optional[dict]:
    """
    Load EXIF data for an image.
    
    Args:
        path: Image file path
        blocking: If True, wait for EXIF to load. If False, return None if not ready.
    
    Returns:
        EXIF dict if available, None if not ready (and blocking=False)
    """
    # Fast path: check cache first
    with self._exif_cache_lock:
        if path in self._exif_cache:
            return self._exif_cache[path]
    
    # Check if already loading
    with self._exif_loading_lock:
        future = self._exif_loading.get(path)
        if future:
            if blocking:
                # Wait for loading to complete
                try:
                    result = future.result(timeout=5.0)
                    return result
                except Exception as e:
                    logging.warning(f"EXIF load failed for {path}: {e}")
                    return {}
            else:
                # Not ready yet, return None
                return None
    
    # Not cached and not loading - trigger background load
    if blocking:
        # Synchronous load (fallback)
        return self._load_exif_sync(path)
    else:
        # Trigger background load
        self._preload_exif_async(path)
        return None  # Not ready yet

def _load_exif_sync(self, path: str) -> dict:
    """Synchronous EXIF loading (original implementation)."""
    try:
        with Image.open(path) as img:
            exif_data = img._getexif() if hasattr(img, "_getexif") and callable(img._getexif) else None
            if not exif_data or not isinstance(exif_data, dict):
                result = {}
            else:
                exif = {}
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, str(tag))
                    exif[tag_name] = value
                result = exif
        
        # Cache the result
        with self._exif_cache_lock:
            self._exif_cache[path] = result
        
        return result
    except Exception as e:
        logging.warning(f"Failed to load EXIF from {path}: {e}")
        result = {}
        with self._exif_cache_lock:
            self._exif_cache[path] = result
        return result

def _preload_exif_async(self, path: str):
    """Trigger background EXIF loading."""
    with self._exif_loading_lock:
        # Check if already loading
        if path in self._exif_loading:
            return
        
        # Submit to thread pool
        future = self._exif_executor.submit(self._load_exif_sync, path)
        self._exif_loading[path] = future
        
        # Clean up future when done
        def cleanup(fut):
            with self._exif_loading_lock:
                self._exif_loading.pop(path, None)
        
        future.add_done_callback(cleanup)
```

### Step 3: Batch Pre-loading on Startup

```python
def preload_exif_batch(self, paths: list[str], priority_paths: Optional[list[str]] = None):
    """
    Pre-load EXIF for multiple images in background.
    
    Args:
        paths: List of image paths to pre-load
        priority_paths: Paths to load first (e.g., visible images)
    """
    # Load priority paths first
    if priority_paths:
        for path in priority_paths:
            if path in paths:
                self._preload_exif_async(path)
        
        # Then load remaining paths
        remaining = [p for p in paths if p not in priority_paths]
        for path in remaining:
            self._preload_exif_async(path)
    else:
        # Load all paths
        for path in paths:
            self._preload_exif_async(path)
```

### Step 4: Integration with ViewModel

```python
# src/viewmodel.py

class PhotoViewModel:
    def load_images(self, directory: str):
        # ... existing code to scan images ...
        
        # After images are loaded, pre-load EXIF in background
        image_paths = [self.model.get_image_path(f) for f in self.images]
        
        # Pre-load EXIF for all images (non-blocking)
        self.model.preload_exif_batch(image_paths)
        
        # ... rest of initialization ...
```

### Step 5: Handle EXIF Not Ready

```python
# src/view.py or src/grouping_service.py

# When EXIF is needed but might not be ready:
exif = model.load_exif(path, blocking=False)
if exif is None:
    # EXIF not ready yet - use defaults or wait
    # Option 1: Use defaults
    exif = {}
    
    # Option 2: Wait for it (if critical)
    exif = model.load_exif(path, blocking=True)
    
    # Option 3: Trigger load and continue (best for UI)
    model._preload_exif_async(path)
    exif = {}  # Use defaults for now
```

## Expected Results

### Before (Current)
- **First access**: 23ms per image (blocking)
- **Total blocking time**: 13.6s for 593 images
- **UI responsiveness**: Blocked during EXIF load

### After (Pre-loaded)
- **Background loading**: Non-blocking, parallel (4 workers)
- **First access**: ~0.1ms (from cache)
- **Total time**: ~3-4s in background (doesn't block UI)
- **UI responsiveness**: No blocking

## Implementation Steps

1. **Add thread-safe cache** to `ImageModel`
2. **Implement async loading** methods
3. **Update `load_exif()`** to be non-blocking by default
4. **Add batch pre-loading** method
5. **Integrate with ViewModel** to pre-load on startup
6. **Update callers** to handle None return value
7. **Test** with real dataset
8. **Measure** performance improvement

## Testing Strategy

1. **Correctness**: Verify EXIF data is same as before
2. **Performance**: Measure UI responsiveness during load
3. **Concurrency**: Test with multiple simultaneous requests
4. **Error handling**: Test with invalid/corrupted images
5. **Memory**: Verify no memory leaks from thread pool

## Rollout Plan

1. **Phase 1**: Implement async loading alongside sync (feature flag)
2. **Phase 2**: Test with small dataset (<100 images)
3. **Phase 3**: Test with full dataset (591 images)
4. **Phase 4**: Enable by default if performance is better
5. **Phase 5**: Remove blocking fallback after validation

## Alternative: Use Existing LazyImageLoader (Recommended)

**Note**: There's already a `LazyImageLoader` class in `src/lazy_loader.py` that handles async loading with thread pool and LRU cache. **This is the recommended approach** - extend it to support batch pre-loading.

### Option: Extend LazyImageLoader

```python
# src/lazy_loader.py

class LazyImageLoader:
    def preload_exif_batch(self, paths: list[str], callback: Optional[Callable] = None):
        """
        Pre-load EXIF for multiple paths in background.
        
        Args:
            paths: List of image paths to pre-load
            callback: Optional callback called for each loaded EXIF (path, exif_dict)
        """
        for path in paths:
            # Use existing async loading infrastructure
            def exif_callback(p, exif):
                if callback:
                    callback(p, exif)
            
            self.get_exif_lazy(path, exif_callback)
    
    def preload_exif_silent(self, paths: list[str]):
        """
        Pre-load EXIF silently (no callbacks, just populate cache).
        """
        for path in paths:
            # Submit to thread pool without callback
            def _load_silent():
                try:
                    if is_cache_disabled():
                        self._load_exif_uncached(path)
                    else:
                        self._cached_exif(path)  # Populates LRU cache
                except Exception:
                    pass  # Silent failure
            
            self.executor.submit(_load_silent)
```

**Advantages**:
- ✅ Reuses existing thread pool and LRU cache
- ✅ No new infrastructure needed
- ✅ Consistent with existing async loading pattern
- ✅ Minimal code changes

**Usage**:
```python
# In PhotoViewModel or PhotoView
self.lazy_loader.preload_exif_silent(image_paths)
```

This is **simpler and recommended** over creating new infrastructure.

