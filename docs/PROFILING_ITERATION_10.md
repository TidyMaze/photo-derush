# Profiling Iteration 10 - 60 Second Run

## Date
2025-12-31

## Profile Duration
60 seconds of normal app usage (loading images, grouping, labeling, retraining)

## Top CPU Hotspots (Cumulative Time)

### 1. Model Inference (sklearn/catboost) - 36.9s
- **Function**: `sklearn.calibration.predict_proba`
- **Time**: 36.941s (900 calls, 0.121s per call)
- **Status**: External library, hard to optimize
- **Note**: Model prediction is inherently expensive

### 2. Torch Serialization - 34.2s
- **Function**: `torch.serialization.persistent_load`
- **Time**: 34.216s (138 calls, 2.444s per call)
- **Status**: YOLO model loading, one-time cost
- **Note**: Already cached, this is during initial load

### 3. PIL Image.open - 14.6s
- **Function**: `PIL.Image.open`
- **Time**: 14.594s (1561 calls, 0.015s per call)
- **Status**: Already optimized with caching
- **Note**: Still significant, but much better than before

### 4. load_exif - 14.6s
- **Function**: `model.load_exif`
- **Time**: 14.590s (15967 calls, 0.001s per call)
- **Status**: ⚠️ **OPTIMIZATION OPPORTUNITY**
- **Note**: Many calls, but EXIF pre-loading already implemented (may not be used yet)

### 5. Signal Emissions - 6.4s
- **Function**: `SignalInstance.emit`
- **Time**: 6.392s (6432 calls, 0.001s per call)
- **Status**: Already optimized with debouncing
- **Note**: Still significant, but improved from before

### 6. _do_refresh_thumbnail_badges - 6.6s
- **Function**: `view._do_refresh_thumbnail_badges`
- **Time**: 6.649s (20 calls, 0.332s per call)
- **Self-time**: 3.176s (48% of cumulative)
- **Status**: ⚠️ **OPTIMIZATION OPPORTUNITY**
- **Note**: Still a hotspot despite previous optimizations

### 7. get_image_timestamp - 6.5s
- **Function**: `viewmodel.get_image_timestamp`
- **Time**: 6.494s (15366 calls, 0.0004s per call)
- **Status**: ⚠️ **OPTIMIZATION OPPORTUNITY**
- **Note**: Very high call count, likely repeated lookups

### 8. _strptime - 5.8s
- **Function**: `_strptime` (datetime parsing)
- **Time**: 5.815s (15768 calls, 0.0004s per call)
- **Status**: ⚠️ **OPTIMIZATION OPPORTUNITY**
- **Note**: Called from `get_image_timestamp`, can be cached

### 9. QWidget.show - 3.9s
- **Function**: `QWidget.show`
- **Time**: 3.926s (4381 calls, 0.001s per call)
- **Self-time**: 2.599s (66% of cumulative)
- **Status**: Qt framework overhead, hard to optimize
- **Note**: Already batched, would need virtual scrolling for more improvement

### 10. catboost._init - 2.9s
- **Function**: `catboost.core._init`
- **Time**: 2.851s (915 calls, 0.003s per call)
- **Self-time**: 2.579s (90% of cumulative)
- **Status**: External library overhead during prediction
- **Note**: Model bundle already cached, this is per-prediction overhead

## Top CPU Hotspots (Self Time)

1. **`_do_refresh_thumbnail_badges`** - 3.176s (20 calls)
2. **`QWidget.show`** - 2.599s (4381 calls)
3. **`catboost._init`** - 2.579s (915 calls)
4. **`_strptime`** - 2.379s (15768 calls)
5. **`dict.get`** - 2.072s (1.1M calls) - Many dictionary lookups
6. **`PIL.TiffImagePlugin.load`** - 1.046s (3058 calls)

## Memory Usage

**Total**: 120.79 MB
- Profiling overhead: ~10 MB (cProfile, tracemalloc)
- Importlib: 52.9 MB (module loading)
- PIL/TIFF: ~1 MB (image metadata)
- Model cache: ~0.9 MB

## Optimization Opportunities

### High Priority

1. **Cache `_strptime` results** (5.8s → ~0.5s expected)
   - **Location**: `viewmodel.get_image_timestamp`
   - **Impact**: 15,768 calls → cached lookups
   - **Effort**: Low (add LRU cache)
   - **Expected**: 90% reduction

2. **Optimize `get_image_timestamp`** (6.5s → ~1s expected)
   - **Location**: `viewmodel.get_image_timestamp`
   - **Impact**: 15,366 calls, many repeated
   - **Effort**: Medium (cache timestamp parsing)
   - **Expected**: 80% reduction

3. **Further optimize `_do_refresh_thumbnail_badges`** (3.2s → ~1.5s expected)
   - **Location**: `view._do_refresh_thumbnail_badges`
   - **Impact**: 20 calls, 0.159s per call
   - **Effort**: Medium (reduce redundant operations)
   - **Expected**: 50% reduction

### Medium Priority

4. **Verify EXIF pre-loading usage** (14.6s → ~3-4s expected)
   - **Location**: `model.load_exif`
   - **Impact**: 15,967 calls (many repeated)
   - **Effort**: Low (ensure pre-loading is called)
   - **Expected**: 70-80% reduction if pre-loading is active

5. **Reduce `dict.get` calls** (2.1s → ~1s expected)
   - **Location**: Various (1.1M calls)
   - **Impact**: Many dictionary lookups
   - **Effort**: Medium (identify hot paths, cache results)
   - **Expected**: 50% reduction in hot paths

### Low Priority

6. **Optimize `PIL.TiffImagePlugin.load`** (1.0s → ~0.5s expected)
   - **Location**: EXIF/TIFF metadata loading
   - **Impact**: 3,058 calls
   - **Effort**: High (requires PIL optimization or caching)
   - **Expected**: 50% reduction with aggressive caching

## Summary

**Total optimizable time**: ~15-20s (out of 52s total)
- High priority: ~10s savings
- Medium priority: ~5s savings
- Low priority: ~1-2s savings

**Remaining hotspots** (hard to optimize):
- Model inference: 36.9s (external library)
- Torch serialization: 34.2s (one-time YOLO load)
- Qt framework: 3.9s (QWidget.show, framework overhead)
- CatBoost overhead: 2.9s (per-prediction initialization)

## Next Steps

1. **Cache `_strptime` results** in `get_image_timestamp`
2. **Add timestamp cache** to `viewmodel`
3. **Profile `_do_refresh_thumbnail_badges`** in detail
4. **Verify EXIF pre-loading** is being used
5. **Identify hot `dict.get` paths** for caching



