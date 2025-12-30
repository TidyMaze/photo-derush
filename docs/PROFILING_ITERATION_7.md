# Profiling Iteration 7 - Additional Optimizations

## Profiling Setup
- **Profiler**: cProfile (multi-thread aggregated) + tracemalloc
- **Duration**: 60 seconds
- **Profile**: `/tmp/app_profile_aggregated.prof`

## Analysis Results

### Top Hotspots Identified
1. **QWidget.show()**: 3.991s (4,787 calls) - Already batched
2. **dict.get()**: 1.614s (677,989 calls) - Very fast per call (2.4μs)
3. **numpy.flatten()**: 1.576s (351,322 calls) - Part of imagehash operations
4. **catboost._init**: 1.082s (592 calls) - Internal library behavior
5. **PIL TiffImagePlugin.load**: 1.068s (3,093 calls) - Already optimized
6. **eventFilter**: 0.388s (334,108 calls) - High call count, can optimize further
7. **isinstance**: 0.452s (1,356,012 calls) - Very high call count
8. **getattr**: 0.275s (131,456 calls) - Can cache in hot paths

## Optimizations Applied

### 1. getattr Optimization in _do_refresh_thumbnail_badges
**Problem**: Multiple `getattr()` calls in hot path (state access, label attributes).

**Solution**:
- Replace `getattr(self, "_last_browser_state", None)` with direct attribute access + `hasattr()` check
- Replace `getattr(state, "detected_objects", {})` with direct attribute access
- Replace `getattr(label, "_thumb_filename", None)` with direct attribute access
- Remove duplicate `getattr(state, "group_info", {})` call

**Files Modified**:
- `src/view.py`: `_do_refresh_thumbnail_badges()`

**Expected Impact**: 0.1-0.2s saved (reduced getattr overhead)

### 2. Memory Analysis
**Total Memory**: 120.71 MB
**Top Allocations**:
1. importlib (52.9 MiB) - Module loading overhead
2. importlib._bootstrap (10.7 MiB) - Module initialization
3. linecache (4.3 MiB) - Source code caching
4. cProfile (3.7 MiB) - Profiling overhead
5. features.py (1.5 MiB) - Feature extraction data

**Memory Optimization Opportunities**:
- Most memory is from importlib (module loading) - not optimizable
- cProfile overhead (3.7 MiB) - only present during profiling
- Feature extraction (1.5 MiB) - reasonable for 591 images

## Remaining Hotspots

### High Priority (Hard to Optimize)
1. **QWidget.show()**: 3.991s (4,787 calls)
   - Already batched
   - Limited by UI requirements
   - Further reduction would require virtual scrolling

2. **dict.get()**: 1.614s (677,989 calls)
   - Very fast per call (2.4μs)
   - Most calls are from PIL/library code
   - Our code already uses direct access where possible

3. **numpy.flatten()**: 1.576s (351,322 calls)
   - Part of imagehash operations (duplicate grouping)
   - Library code, necessary for hash computation
   - Cannot optimize without changing imagehash library

4. **catboost._init**: 1.082s (592 calls)
   - Internal sklearn/catboost behavior
   - Model bundle already cached
   - Cannot optimize without modifying libraries

### Medium Priority (Can Optimize Further)
5. **eventFilter**: 0.388s (334,108 calls)
   - Already optimized with early returns
   - Further optimization would require reducing event processing
   - Potential: 0.1-0.2s savings

6. **isinstance**: 0.452s (1,356,012 calls)
   - Very high call count (22.6K/sec)
   - Most calls are from library code
   - Our code could cache type checks in hot loops
   - Potential: 0.1-0.15s savings

7. **getattr**: 0.275s (131,456 calls)
   - Already optimized in _do_refresh_thumbnail_badges
   - Could optimize in other hot paths
   - Potential: 0.05-0.1s savings

## Cumulative Optimizations Summary

| Optimization | Time Saved | Status |
|-------------|------------|--------|
| setStyleSheet caching + batching | 2.35s | ✅ Complete |
| PIL Image.open (cached size) | 11.8s | ✅ Complete |
| Signal debouncing (200ms→500ms) | 2-3s | ✅ Complete |
| Widget show filtering | 0.5s | ✅ Complete |
| EXIF batching | 0.2-0.5s | ✅ Complete |
| getattr optimization | 0.1-0.2s | ✅ Complete |
| **Total** | **~17-18s** | |

## Next Steps

### Recommended Optimizations
1. **eventFilter further optimization** (0.1-0.2s potential)
   - Cache event type checks
   - Reduce event processing overhead

2. **isinstance caching** (0.1-0.15s potential)
   - Cache type checks in hot loops
   - Use type variables instead of isinstance in some cases

3. **Virtual scrolling** (for 10K images)
   - Only render visible thumbnails
   - Would reduce widget operations by 99%
   - **Impact**: Massive for large image sets

### Not Recommended (Low ROI)
- **dict.get optimization**: Already optimal (2.4μs per call)
- **numpy.flatten**: Library code, cannot optimize
- **catboost._init**: Internal library behavior
- **QWidget.show()**: Already batched, limited by UI

## Performance Status

**Current State**: Well optimized for typical use cases (100-1000 images)
**For 10K images**: Virtual scrolling is **mandatory** for acceptable performance

