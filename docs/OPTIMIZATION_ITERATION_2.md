# Optimization Iteration 2 - Profile Analysis

## Profile Results (60s run, 591 images)

### Top Hotspots (Self Time)
1. **tracemalloc._group_by**: 4.324s (profiling overhead, ignore)
2. **QWidget.show**: 2.453s (2652 calls) - Still high despite batching
3. **PIL TiffImagePlugin.load**: 0.928s (2395 calls) - EXIF loading
4. **imagehash.__sub__**: 0.835s (174,770 calls) - Duplicate grouping O(n²) comparisons
5. **PIL TiffImagePlugin._setitem**: 0.729s (38,885 calls) - EXIF processing
6. **group_near_duplicates**: 0.432s (1 call) - Improved from previous iterations
7. **hex_to_hash**: 0.279s (1441 calls) - Still some conversions

### Top Hotspots (Cumulative Time)
1. **load_exif**: 13.707s (1775 calls) - EXIF loading dominates
2. **compute_grouping**: 13.079s - Overall grouping computation
3. **predict_proba**: 11.126s - ML predictions
4. **group_near_duplicates**: 2.647s - Duplicate grouping

## Analysis

### Improvements from Iteration 1
- ✅ Badge refresh optimization (visible thumbnails only) - working
- ✅ Duplicate grouping hash object caching - working (reduced hex_to_hash from 76k to 1.4k calls)

### Remaining Issues
1. **O(n²) duplicate grouping**: 174k `imagehash.__sub__` calls show we're still doing O(n²) comparisons
   - For 591 images: ~174k comparisons = 591 * 591 / 2 (matches profile)
   - Need: LSH or spatial index to reduce to O(n log n)

2. **QWidget.show calls**: 2652 calls (2.453s)
   - Batching is working but still too many calls
   - Some direct show() calls bypass batching

3. **EXIF loading**: 0.928s + 0.729s = 1.657s total
   - Already cached but still significant
   - PIL TiffImagePlugin operations are expensive

## Next Optimizations

### Priority 1: Reduce QWidget.show calls
- Remove direct `show()` calls that bypass batching
- Only show widgets that actually changed visibility
- Cache visibility state to avoid redundant show/hide

### Priority 2: Optimize duplicate grouping further
- Current: O(n²) with early termination
- Option A: Use LSH (Locality-Sensitive Hashing) - O(n log n)
- Option B: Spatial index (KD-tree, ball tree) - O(n log n)
- Option C: Pre-filter by hash buckets - reduce comparisons

### Priority 3: EXIF loading optimization
- Already cached, but PIL operations are still expensive
- Consider lazy EXIF loading (only when needed)
- Batch EXIF reads if possible

## Performance Targets

| Metric | Current | Target | Strategy |
|--------|---------|--------|----------|
| QWidget.show | 2.453s | <1s | Remove direct calls, cache visibility |
| imagehash.__sub__ | 0.835s (174k) | <0.2s | LSH or spatial index |
| EXIF loading | 1.657s | <0.5s | Lazy loading, batch reads |
| Total runtime | 34.123s | <25s | Combined optimizations |

