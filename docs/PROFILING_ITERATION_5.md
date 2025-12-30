# Profiling Iteration 5 - Continued Optimization

## Profiling Setup
- **Profiler**: cProfile (main thread) + tracemalloc (memory, all threads)
- **Duration**: 60 seconds
- **Note**: Using cProfile + tracemalloc (no sudo required)

## CPU Hotspots Identified

### Top Functions by Self Time
1. **tracemalloc._group_by**: 4.376s (profiling overhead)
2. **QWidget.show()**: 3.413s (4316 calls) - widget visibility
3. **catboost._init**: 1.060s (592 calls) - model initialization overhead
4. **PIL TiffImagePlugin.load**: 1.032s (3018 calls) - image loading (reduced from 12.8s!)
5. **dict.get**: 1.016s (668573 calls) - dict lookups (very fast per call)
6. **imagehash.__sub__**: 0.840s (174770 calls) - hash comparisons
7. **numpy.ndarray.flatten**: 0.763s (351316 calls) - array operations
8. **PIL TiffImagePlugin._setitem**: 0.676s (40437 calls) - image metadata
9. **builtins.hash**: 0.648s (2848313 calls) - hash operations
10. **group_near_duplicates**: 0.577s (1 call) - grouping (already optimized)
11. **_do_refresh_thumbnail_badges**: 0.494s (9 calls) - badge refresh
12. **eventFilter**: 0.331s (285352 calls) - event filtering

## Memory Hotspots Identified

### Top Allocations
1. **importlib/bootstrap**: 52.9 MiB (module loading - unavoidable)
2. **importlib/bootstrap**: 10.7 MiB (module loading)
3. **linecache**: 5.4 MiB (source code caching)
4. **features.py**: 1.5 MiB (feature extraction)
5. **model.py**: 861 KiB (EXIF loading)

## Optimizations Applied (Previous Iterations)

1. ✅ Signal debouncing: 200ms → 500ms
2. ✅ EXIF batching: batch loading in groups of 50
3. ✅ Widget show filtering: only process widgets that need state change
4. ✅ PIL Image.open caching: store original size in thumbnail metadata

## Remaining Hotspots

### High Priority
1. **catboost._init** (1.060s, 592 calls)
   - **Issue**: Model initialization overhead on each prediction
   - **Solution**: Cache/reuse model instances, batch predictions
   - **Potential savings**: 0.8-1.0s

2. **eventFilter** (0.331s, 285352 calls)
   - **Issue**: Very high call count (285K calls in 60s = ~4.7K/sec)
   - **Solution**: Further optimize early returns, reduce event processing
   - **Potential savings**: 0.2-0.3s

3. **QWidget.show()** (3.413s, 4316 calls)
   - **Issue**: Still many show() calls despite batching
   - **Solution**: Further reduce redundant visibility changes
   - **Potential savings**: 0.5-1.0s (limited by UI requirements)

### Medium Priority
4. **PIL TiffImagePlugin operations** (1.032s + 0.676s = 1.7s total)
   - **Issue**: Image loading and metadata operations
   - **Solution**: Already optimized, but could cache more aggressively
   - **Potential savings**: 0.2-0.4s

5. **dict.get** (1.016s, 668573 calls)
   - **Issue**: Many dict lookups (very fast per call: 1.5μs)
   - **Solution**: Already optimal, but could reduce lookup count
   - **Potential savings**: 0.1-0.2s

## Next Steps
1. Optimize catboost model initialization (cache/reuse instances)
2. Further optimize eventFilter (reduce call overhead)
3. Batch catboost predictions to reduce initialization overhead
4. Consider lazy loading for less critical UI updates

