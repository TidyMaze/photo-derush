# Hotspots & Optimization Strategies

## Profile Analysis Summary

### With Caches Enabled (2 min run, 591 images)
- **Total runtime**: 30.2 seconds
- **Top bottleneck**: Duplicate grouping (17.4s cumulative)
- **Memory**: 9.97 MB

### With Caches Disabled (Cold Cache)
- **Total runtime**: ~180 seconds (estimated)
- **Top bottleneck**: Model loading (204s cumulative)
- **Memory**: 39.28 MB

---

## 🔥 Critical Hotspots

### 1. Duplicate Grouping (Perceptual Hashing) - 17.4s ⚠️ **BIGGEST BOTTLENECK**

**Location**: 
- `imagehash/__init__.py:166(hex_to_hash)`: 14.9s total, 76,215 calls
- `photo_grouping.py:121(group_near_duplicates)`: 17.4s cumulative
- `imagehash/__init__.py:106(__sub__)`: 0.4s total, 76,068 calls (Hamming distance)

**Problem**:
- Computing perceptual hashes for all images (591 images → 76k hash operations)
- O(n²) comparison: comparing every hash against every other hash
- `hex_to_hash()` called 76,215 times (converting string to hash object)

**Current Implementation**:
```python
# In photo_grouping.py:121
# Builds graph with O(n²) comparisons
for existing_hash_str, gid in hash_to_group.items():
    existing_hash = imagehash.hex_to_hash(existing_hash_str)  # SLOW!
    if phash - existing_hash <= hamming_threshold:
        ...
```

**Optimizations**:

#### A. Cache Hash Objects (Not Strings) ⭐⭐⭐
**Impact**: 50-70% reduction (14.9s → 5-7s)
**Effort**: Low
```python
# Instead of storing hash strings and converting back:
hash_to_group: dict[imagehash.ImageHash, int] = {}  # Store hash objects directly
# Eliminates 76k hex_to_hash() calls
```

#### B. Use Locality-Sensitive Hashing (LSH) ⭐⭐⭐
**Impact**: O(n²) → O(n log n) or O(n)
**Effort**: Medium
- Use LSH to find similar hashes without comparing all pairs
- Implement approximate nearest neighbor search
- Libraries: `annoy`, `faiss`, or custom LSH

#### C. Parallel Hash Computation ⭐⭐
**Impact**: 2-4x speedup on multi-core
**Effort**: Low
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    hashes = list(executor.map(compute_hash, filenames))
```

#### D. Skip Grouping for Small Datasets ⭐
**Impact**: Immediate skip for < 20 images
**Effort**: Already implemented (line 123 in duplicate_grouping.py)

#### E. Incremental Grouping ⭐⭐
**Impact**: Faster initial display, background completion
**Effort**: Medium
- Group visible images first
- Continue grouping in background thread
- Update UI as groups are discovered

**Combined Impact**: 17.4s → **2-4s** (75-85% improvement)

---

### 2. Event Filter - 1.6s (Warm Cache) / 1.47s (Cold Cache)

**Location**: `view.py:3487(eventFilter)`
- **Calls**: 158,486 (warm), 146k (cold)
- **Time**: 1.6s total, 10μs per call

**Problem**:
- Called for every Qt event (mouse, keyboard, paint, etc.)
- Even small overhead adds up with 150k+ calls

**Current Code**:
```python
def eventFilter(self, obj, event):
    # Called 158k times
    if event.type() == QEvent.Type.MouseMove:
        # Handle mouse move
    ...
```

**Optimizations**:

#### A. Early Return for Unhandled Events ⭐⭐⭐
**Impact**: 30-50% reduction (1.6s → 0.8-1.1s)
**Effort**: Low
```python
def eventFilter(self, obj, event):
    event_type = event.type()
    # Fast path: return False immediately for unhandled types
    if event_type not in (QEvent.Type.MouseMove, QEvent.Type.Enter, QEvent.Type.Leave):
        return False
    # ... handle specific events
```

#### B. Cache Event Type Checks ⭐⭐
**Impact**: 10-20% reduction
**Effort**: Low
```python
# Cache common event types as constants
_MOUSE_MOVE = QEvent.Type.MouseMove
_ENTER = QEvent.Type.Enter
# Use cached constants instead of enum lookups
```

#### C. Reduce Event Handler Overhead ⭐
**Impact**: 5-10% reduction
**Effort**: Low
- Minimize function calls in hot path
- Use local variables instead of attribute lookups
- Avoid logging in event filter

**Combined Impact**: 1.6s → **0.7-1.0s** (30-50% improvement)

---

### 3. Thumbnail Loading - 4.5s (Warm Cache) / 7.7s (Cold Cache)

**Location**: 
- `view.py:2690(_on_thumbnail_loaded)`: 0.42s total, 591 calls (warm)
- `PIL/JpegImagePlugin.py:384(load_read)`: 7.7s cumulative (cold)

**Problem**:
- Loading 591 thumbnails synchronously
- JPEG decoding is CPU-intensive
- File I/O blocking UI thread

**Optimizations**:

#### A. Virtual Scrolling ⭐⭐⭐ **CRITICAL FOR 10K IMAGES**
**Impact**: 90% reduction (591 → ~50-100 visible)
**Effort**: Medium
- Only render thumbnails in viewport
- Reuse thumbnail widgets as user scrolls
- **For 10K images**: Load 50-100 instead of 10,000

#### B. Lazy Loading (Already Partially Implemented) ⭐⭐⭐
**Impact**: 80% faster initial display
**Effort**: Low
- Load thumbnails on-demand as user scrolls
- Prefetch 1-2 screens ahead

#### C. Parallel Thumbnail Generation ⭐⭐
**Impact**: 2-4x speedup on multi-core
**Effort**: Medium
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    thumbnails = list(executor.map(load_thumbnail, image_paths))
```

#### D. Lower Quality Initial Thumbnails ⭐
**Impact**: 30-50% faster loading
**Effort**: Low
- Load 128x128 initially, upgrade to 256x256 on demand
- Progressive JPEG loading

**Combined Impact**: 4.5s → **0.5-1.0s** (80-90% improvement)

---

### 4. Model Loading (Cold Cache Only) - 204s

**Location**: `torch/serialization.py:2068(persistent_load)`
- **Impact**: 204s cumulative (one-time cost)
- **Calls**: Model loaded 21 times (but should be cached)

**Problem**:
- YOLOv8 weights loaded from disk every time
- Even with caching, first load is expensive

**Optimizations**:

#### A. Pre-load Model on Startup ⭐⭐⭐
**Impact**: Eliminates 204s delay when first detection runs
**Effort**: Low
```python
# In app.py or viewmodel.py startup
from src.object_detection import _load_model
_load_model("auto")  # Pre-load in background thread
```

#### B. Model Caching (Already Implemented) ⭐⭐⭐
**Impact**: Prevents reloads after first load
**Status**: ✅ Already done in `object_detection.py:512-518`

#### C. Use Lighter Model Variant ⭐
**Impact**: 50% faster loading (yolov8n → yolov8n-tiny if available)
**Effort**: Low (if tiny variant exists)

**Combined Impact**: 204s → **0s** (after first load)

---

### 5. Object Detection Inference - 9.2s (Cold Cache)

**Location**: `ultralytics/engine/predictor.py`, `object_detection.py:596`

**Problem**:
- Processing images one-by-one
- YOLOv8 supports batch prediction but not used

**Optimizations**:

#### A. Batch Detection ⭐⭐⭐
**Impact**: 2-3x faster (process 8-16 images at once)
**Effort**: Medium
```python
# YOLOv8 supports batch prediction
results = model.predict(images_batch, imgsz=640, conf=0.25)
```

#### B. Lazy Detection ⭐⭐
**Impact**: Faster initial display
**Effort**: Medium
- Only detect objects for visible images
- Detect in background as user scrolls

#### C. Skip Detection for Already Processed ⭐
**Impact**: Eliminates redundant work
**Effort**: Low (cache already exists)

**Combined Impact**: 9.2s → **3-4s** (60% improvement)

---

### 6. File I/O Operations - 2.1s (Cold Cache)

**Location**: 
- `{method 'read' of '_io.BufferedReader' objects}`: 2.1s, 100k calls
- `posix.lstat`: 0.24s, 5k calls

**Problem**:
- Many small file reads
- Redundant stat calls

**Optimizations**:

#### A. Batch File Operations ⭐⭐
**Impact**: 20-30% reduction
**Effort**: Medium
- Group file reads together
- Use readahead when possible

#### B. Cache File Stats (Already Implemented) ⭐
**Impact**: Eliminates redundant stat calls
**Status**: ✅ Already done

#### C. Async I/O ⭐⭐
**Impact**: Non-blocking UI
**Effort**: Medium
- Use `asyncio` or thread pool for file I/O
- Don't block UI thread

**Combined Impact**: 2.1s → **1.4-1.6s** (25-35% improvement)

---

### 7. Thumbnail Badge Refresh - 2.07s (Cold Cache)

**Location**: `view.py:1883(_refresh_thumbnail_badges)`
- **Calls**: 41 calls, 50ms per call

**Problem**:
- Refreshing badges on all thumbnails
- Even when nothing changed

**Optimizations**:

#### A. Only Refresh Visible Thumbnails ⭐⭐⭐
**Impact**: 80-90% reduction (41 → 4-8 visible)
**Effort**: Medium
- Track visible thumbnail range
- Only update visible badges

#### B. Debounce Badge Updates ⭐⭐
**Impact**: Reduces redundant refreshes
**Effort**: Low (already partially done with timer)

#### C. Batch Badge Updates ⭐
**Impact**: 10-20% reduction
**Effort**: Low
- Update multiple badges in single paint event

**Combined Impact**: 2.07s → **0.2-0.4s** (80-90% improvement)

---

## 📊 Performance Targets

### Current Performance (591 images)

| Operation | Cold Cache | Warm Cache | Target |
|-----------|------------|------------|--------|
| Initial Load | ~180s | 30s | < 5s |
| Duplicate Grouping | N/A | 17.4s | < 3s |
| Thumbnail Loading | 7.7s | 4.5s | < 1s |
| Object Detection | 9.2s | 0s (cached) | < 3s |
| Event Filter | 1.47s | 1.6s | < 0.8s |

### Target Performance (10K images)

| Operation | Current (Est.) | Target | Strategy |
|-----------|----------------|--------|----------|
| Initial Load | 3000s (50 min) | < 10s | Virtual scrolling |
| Duplicate Grouping | 300s | < 30s | LSH + parallel |
| Thumbnail Loading | 130s | < 2s | Virtual scrolling |
| Object Detection | 150s | Lazy | On-demand only |
| Event Filter | 25s | < 5s | Optimize hot path |

---

## 🎯 Implementation Priority

### P0 (Critical - Do First)
1. **Virtual Scrolling** ⭐⭐⭐
   - Impact: 90% reduction in thumbnail work
   - Effort: Medium
   - **Required for 10K images**

2. **Optimize Duplicate Grouping** ⭐⭐⭐
   - Cache hash objects (not strings)
   - Use LSH for O(n log n) instead of O(n²)
   - Impact: 17.4s → 2-4s

3. **Pre-load YOLOv8 Model** ⭐⭐⭐
   - Impact: Eliminates 204s delay
   - Effort: Low

### P1 (High Impact)
4. **Batch Object Detection** ⭐⭐
   - Impact: 2-3x faster detection
   - Effort: Medium

5. **Optimize Event Filter** ⭐⭐
   - Early return for unhandled events
   - Impact: 1.6s → 0.8s

6. **Parallel Hash Computation** ⭐⭐
   - Impact: 2-4x speedup
   - Effort: Low

### P2 (Medium Impact)
7. **Lazy Detection** ⭐⭐
   - Only detect visible images
   - Effort: Medium

8. **Batch Badge Updates** ⭐
   - Only refresh visible thumbnails
   - Effort: Medium

9. **Async I/O** ⭐⭐
   - Non-blocking file operations
   - Effort: Medium

### P3 (Nice-to-Have)
10. **Progressive Thumbnails** ⭐
    - Lower quality initially
    - Effort: Low

11. **Background Cache Warming** ⭐
    - Pre-compute in background
    - Effort: Low

---

## 🚀 Quick Wins (Low Effort, High Impact)

1. **Cache Hash Objects** (5 min)
   - Change `dict[str, int]` to `dict[imagehash.ImageHash, int]`
   - Impact: 14.9s → 5-7s

2. **Early Return in Event Filter** (10 min)
   - Add early return for unhandled events
   - Impact: 1.6s → 0.8s

3. **Pre-load Model** (5 min)
   - Call `_load_model()` on startup
   - Impact: Eliminates 204s delay

4. **Parallel Hash Computation** (15 min)
   - Use ThreadPoolExecutor
   - Impact: 2-4x speedup

**Total Quick Wins Impact**: ~20-25s saved (warm cache) or ~220s saved (cold cache)

---

## 📝 Notes

- **Cold cache** performance is dominated by model loading (204s) and I/O (7.7s JPEG loading)
- **Warm cache** performance is dominated by duplicate grouping (17.4s)
- **For 10K images**: Virtual scrolling is **mandatory** (cannot load 10K thumbnails)
- Most optimizations work for both cold and warm cache scenarios
- Memory usage is acceptable (9.97 MB warm, 39 MB cold)

