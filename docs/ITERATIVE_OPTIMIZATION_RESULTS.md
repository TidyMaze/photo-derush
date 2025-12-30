# Iterative Optimization Results

## Summary

Performed iterative profiling and optimization, identifying and fixing hotspots one by one.

## Baseline (Iteration 1)

**Total Runtime**: 21.82 seconds

**Top Hotspots**:
1. PIL TiffImagePlugin.load: **2.545s** (EXIF metadata loading)
2. QWidget.show: 2.498s (UI rendering)
3. PIL TiffImagePlugin._setitem: 1.839s (EXIF processing)
4. dict.get: 0.888s (dictionary lookups)
5. imagehash.__sub__: 0.825s (Hamming distance)

---

## Iteration 2: EXIF Cache ⭐⭐⭐

**Optimization**: Added in-memory cache to `ImageModel.load_exif()` to avoid repeated file opens.

**Changes**:
- Added `_exif_cache` dictionary to cache EXIF data per path
- Cache persists for the lifetime of the ImageModel instance
- Even empty results are cached to avoid retrying failed loads

**Results**:
- **Total Runtime**: 16.23s (**26% faster!**)
- **PIL TiffImagePlugin.load**: 2.545s → 0.846s (**67% reduction**)
- **Function calls**: 10.1M → 6.1M (40% reduction)

**Impact**: Eliminated 1.7 seconds of redundant EXIF file operations.

---

## Iteration 3: Widget Show Optimization (Attempted)

**Optimization**: Added `isVisible()` checks before calling `show()` to avoid redundant show operations.

**Changes**:
- Check `widget.isVisible()` before calling `widget.show()`
- Applied to labels, badge overlays, bbox overlays, group badge overlays

**Results**:
- **Total Runtime**: 16.31s (slightly worse)
- **QWidget.show**: 2.399s → 2.588s (no improvement)
- The `isVisible()` check added overhead without benefit

**Conclusion**: Reverted - `isVisible()` check is slower than just calling `show()`.

---

## Final State (After Iteration 2)

**Total Runtime**: 16.23 seconds (**25% faster than baseline**)

**Remaining Hotspots**:
1. **QWidget.show**: 2.399s (Qt internal rendering - hard to optimize)
2. **imagehash.__sub__**: 0.821s (Hamming distance - core algorithm)
3. **PIL TiffImagePlugin.load**: 0.846s (remaining EXIF loads - already optimized)
4. **numpy.flatten**: 0.739s (part of imagehash operations)
5. **dict.get**: 0.685s (mostly from PIL library code)

---

## Optimization Impact Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| **Total Runtime** | 21.82s | 16.23s | **25% faster** |
| **EXIF Loading** | 2.545s | 0.846s | **67% reduction** |
| **Function Calls** | 10.1M | 6.1M | **40% reduction** |

---

## Why Further Optimizations Are Limited

### 1. QWidget.show (2.4s)
- **Issue**: Qt's internal widget rendering
- **Why hard to optimize**: Core Qt functionality, called for every visible widget
- **Potential**: Virtual scrolling could reduce number of visible widgets (for 10K images)

### 2. imagehash.__sub__ (0.82s)
- **Issue**: Hamming distance calculations for duplicate grouping
- **Why hard to optimize**: Core algorithm, O(n²) comparisons
- **Potential**: LSH (Locality-Sensitive Hashing) could reduce to O(n log n), but requires algorithm change

### 3. PIL TiffImagePlugin.load (0.85s)
- **Issue**: Remaining EXIF metadata loads
- **Why hard to optimize**: Already cached, remaining calls are from feature extraction
- **Potential**: Could cache EXIF in feature extraction too, but less impactful

### 4. numpy.flatten (0.74s)
- **Issue**: Part of imagehash operations
- **Why hard to optimize**: Library code, necessary for hash computation
- **Potential**: None without changing imagehash library

### 5. dict.get (0.69s)
- **Issue**: Dictionary lookups (mostly from PIL)
- **Why hard to optimize**: Most calls are from PIL library code
- **Potential**: Limited - our code already uses direct access where possible

---

## Recommendations for 10K Images

For scaling to 10K images, the following optimizations are **critical**:

### 1. Virtual Scrolling ⭐⭐⭐ **MANDATORY**
- Only render visible thumbnails (50-100 instead of 10,000)
- **Impact**: 99% reduction in widget operations
- **Effort**: Medium

### 2. LSH for Duplicate Grouping ⭐⭐⭐
- Replace O(n²) comparisons with LSH
- **Impact**: O(n²) → O(n log n) or O(n)
- **Effort**: High (algorithm change)

### 3. Progressive Loading ⭐⭐
- Load images in chunks of 100-200
- **Impact**: Faster initial display
- **Effort**: Medium

### 4. Background Processing ⭐⭐
- Move feature extraction, grouping, prediction to background threads
- **Impact**: Non-blocking UI
- **Effort**: Medium

---

## Conclusion

**Achieved**: 25% performance improvement through EXIF caching

**Remaining bottlenecks** are primarily:
- Qt internal operations (rendering)
- Core algorithms (duplicate grouping)
- Library code (PIL, numpy)

**For 10K images**: Virtual scrolling is **mandatory** - cannot render 10K widgets efficiently.

**Next steps**: Implement virtual scrolling for 10K image support.

