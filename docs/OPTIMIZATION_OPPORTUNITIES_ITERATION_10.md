# Optimization Opportunities - Iteration 10

## Summary

After 60 seconds of profiling, identified **~15-20s of optimizable time** out of 52s total CPU time.

## Top 3 Optimization Opportunities

### 1. Cache `get_image_timestamp` Results ⭐⭐⭐
**Impact**: 6.5s → ~0.5s (90% reduction)

**Problem**:
- Called **15,366 times** in 60 seconds
- Each call does:
  - `load_exif()` (expensive, 15,967 total calls)
  - `datetime.strptime()` (expensive, 15,768 total calls)
  - `os.path.getmtime()` (file system access)
- **No caching** - same images parsed repeatedly

**Solution**:
```python
# In viewmodel.py, add LRU cache to get_image_timestamp
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_image_timestamp_cached(fname: str) -> datetime:
    # ... existing logic ...
```

**Expected Impact**:
- 15,366 calls → ~591 unique images (cached)
- 5.8s `_strptime` → ~0.1s (cached lookups)
- 14.6s `load_exif` → ~1-2s (cached EXIF)
- **Total: 6.5s → ~0.5s**

**Effort**: Low (add decorator, ensure cache invalidation)

---

### 2. Further Optimize `_do_refresh_thumbnail_badges` ⭐⭐
**Impact**: 3.2s → ~1.5s (50% reduction)

**Problem**:
- Called **20 times** in 60 seconds
- **3.176s self-time** (48% of 6.6s cumulative)
- Still doing redundant operations

**Current Optimizations**:
- ✅ Cached state access
- ✅ Batched visibility changes
- ✅ Debounced calls

**Remaining Issues**:
- Likely redundant repaints
- Possible repeated dictionary lookups
- Widget updates could be batched further

**Solution**:
- Profile function in detail to identify hot paths
- Cache more intermediate results
- Reduce widget update frequency

**Expected Impact**: 3.2s → ~1.5s

**Effort**: Medium (requires detailed profiling)

---

### 3. Verify/Enable EXIF Pre-loading ⭐⭐
**Impact**: 14.6s → ~3-4s (70-80% reduction)

**Problem**:
- `load_exif` called **15,967 times** in 60 seconds
- EXIF pre-loading was implemented but may not be used
- Many repeated calls for same images

**Solution**:
- Verify `LazyImageLoader.preload_exif_silent()` is called on startup
- Ensure EXIF is pre-loaded before `get_image_timestamp` is called
- Add logging to verify pre-loading is active

**Expected Impact**: 14.6s → ~3-4s (if pre-loading is active)

**Effort**: Low (verify integration)

---

## Additional Opportunities

### 4. Reduce `dict.get` Calls (2.1s)
**Impact**: 2.1s → ~1s (50% reduction)

**Problem**:
- **1.1M `dict.get` calls** in 60 seconds
- Many repeated lookups in hot paths

**Solution**:
- Identify hot paths with many `dict.get` calls
- Cache frequently accessed values
- Use direct attribute access where possible

**Effort**: Medium (requires profiling hot paths)

### 5. Optimize `PIL.TiffImagePlugin.load` (1.0s)
**Impact**: 1.0s → ~0.5s (50% reduction)

**Problem**:
- **3,058 calls** to TIFF loading
- Part of EXIF metadata extraction

**Solution**:
- Aggressive caching of TIFF metadata
- Pre-load TIFF data with EXIF

**Effort**: High (requires PIL optimization)

---

## Hard to Optimize (External/Framework)

1. **Model Inference** (36.9s) - sklearn/catboost, external library
2. **Torch Serialization** (34.2s) - YOLO model loading, one-time cost
3. **Qt Framework** (3.9s) - `QWidget.show`, framework overhead
4. **CatBoost Overhead** (2.9s) - per-prediction initialization

---

## Recommended Implementation Order

1. **Cache `get_image_timestamp`** (Low effort, High impact: 6.5s → 0.5s)
2. **Verify EXIF pre-loading** (Low effort, High impact: 14.6s → 3-4s)
3. **Optimize `_do_refresh_thumbnail_badges`** (Medium effort, Medium impact: 3.2s → 1.5s)
4. **Reduce `dict.get` calls** (Medium effort, Low impact: 2.1s → 1s)

**Total Expected Savings**: ~15-20s (out of 52s total)

---

## Next Steps

1. Add `@lru_cache` to `get_image_timestamp`
2. Verify EXIF pre-loading is called in `PhotoViewModel.load_images()`
3. Profile `_do_refresh_thumbnail_badges` in detail
4. Run profiler again to measure improvements


