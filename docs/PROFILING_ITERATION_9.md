# Profiling Iteration 9 - hasattr() Optimization

## Date
2025-12-31

## CPU Profile Analysis

### Top Hotspots Identified
1. **`imagehash.__sub__`**: 3.932s (85191 calls) - Hash comparisons (already partially optimized)
2. **`hasattr()`**: 3.284s (180228 calls) - Attribute existence checks ⚠️ **OPTIMIZATION TARGET**
3. **`load_exif`**: 13.740s cumulative (593 calls) - EXIF loading (already cached)
4. **`SignalInstance.emit()`**: 8.359s cumulative (4249 calls) - Signal emissions
5. **`PIL Image.open`**: 6.039s cumulative (1239 calls) - Image loading (already cached)

## Memory Profile Analysis

### Top Allocations
1. **importlib**: 52.9 MiB + 10.7 MiB = 63.6 MiB (module loading - unavoidable)
2. **cProfile**: 3.7 MiB (profiling overhead - only during profiling)
3. **features.py**: 1.5 MiB (feature extraction)
4. **model.py**: 861 KiB (EXIF loading)
5. **PIL TiffImagePlugin**: 1.4 MiB (image metadata)

**Total Memory**: 120.55 MB (reasonable for 591 images)

## Optimizations Applied

### 1. Optimized `hasattr()` Calls (3.284s → expected ~1.5-2s)
**Problem**: 180,228 `hasattr()` calls consuming 3.284s CPU time.

**Solution**: Replace `hasattr()` with faster alternatives:
- **Use `getattr()` with default**: For optional attributes, `getattr(obj, "attr", default)` is faster than `hasattr()` + attribute access
- **Direct access with try/except**: For attributes that are always set, use direct access
- **Cache results**: For attributes that don't change, cache the result

**Files Modified**:
- `src/view.py`: 
  - `_do_refresh_thumbnail_badges()`: Replaced `hasattr(self, "_last_browser_state")` with `getattr()`
  - Replaced `hasattr(state, "detected_objects")` with `getattr()`
  - Replaced `hasattr(label, "_thumb_filename")` with try/except (always set)
  - Replaced `hasattr(label, "_badge_painted")` with `getattr()`
  - Replaced `hasattr(self.viewmodel, "selection_model")` with `getattr()`
  - Replaced `hasattr(self, "selected_filenames")` with direct access (always set)

**Expected Impact**: 
- 40-50% reduction in `hasattr()` overhead (3.284s → ~1.5-2s)
- Faster attribute access in hot loops

## Remaining Optimization Opportunities

### High Priority
1. **`imagehash.__sub__`** (3.932s, 85191 calls)
   - Already partially optimized (hash objects cached)
   - Could use LSH (Locality-Sensitive Hashing) for O(n²) → O(n log n)
   - **Effort**: Medium, **Impact**: High

2. **`load_exif`** (13.740s cumulative, 593 calls)
   - Already cached, but first call per image is slow
   - Could pre-load EXIF in background thread
   - **Effort**: Low, **Impact**: Medium

3. **`SignalInstance.emit()`** (8.359s cumulative, 4249 calls)
   - Already debounced in taskrunner
   - Could batch more signal emissions
   - **Effort**: Medium, **Impact**: Medium

### Medium Priority
4. **`PIL Image.open`** (6.039s cumulative, 1239 calls)
   - Already cached in some places
   - Could improve cache hit rate
   - **Effort**: Low, **Impact**: Low-Medium

5. **Memory allocations** (120.55 MB total)
   - Most memory is from importlib (unavoidable)
   - Feature extraction (1.5 MiB) is reasonable
   - **Effort**: Low, **Impact**: Low

## Performance Summary

### Before (Iteration 8)
- `hasattr()`: 3.284s (180228 calls)
- Total runtime: ~30s

### After (Iteration 9)
- `hasattr()`: Expected ~1.5-2s (reduced by 40-50%)
- Total runtime: Expected ~28-29s (1-2s improvement)

## Next Steps

1. Profile again to measure actual `hasattr()` reduction
2. Consider LSH for imagehash comparisons if grouping is still slow
3. Pre-load EXIF data in background thread for faster initial access
4. Further optimize signal emissions if still a bottleneck



