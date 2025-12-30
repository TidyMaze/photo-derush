# Optimization Solutions for Identified Hotspots

## YOLOv8 Loading Analysis

**Finding**: `_load_model` called **21 times** (0.028s total), but `persistent_load` shows **204s cumulative** time.

**Root Cause**: 
- `YOLO(model_path_or_name)` constructor loads weights from disk every time
- Even though model object is cached, the `YOLO()` call inside `YOLOv8Wrapper.__init__` happens before caching check
- Torch serialization (`persistent_load`) loads model weights from `.pt` file repeatedly

**Solution**: Pre-instantiate YOLO model once and reuse it.

---

## Solutions by Hotspot

### 1. YOLOv8 Model Loading (204s → <1s)

**Problem**: Model weights loaded from disk multiple times.

**Solution**:
```python
# In object_detection.py, modify YOLOv8Wrapper to reuse YOLO instance
# Move YOLO instantiation outside wrapper, cache it globally
```

**Implementation**:
- Create global `_yolo_instance` that's reused
- Only instantiate `YOLO()` once per process
- Cache the YOLO object, not just the wrapper

**Expected Impact**: 204s → <1s (one-time load)

---

### 2. Badge Refresh (2.07s, 50ms/call × 41 calls)

**Problem**: `_refresh_thumbnail_badges` called 41 times, 50ms each.

**Solution**:
- Debounce badge refresh operations
- Batch updates (collect changes, apply once)
- Only refresh visible thumbnails

**Implementation**:
```python
# In view.py
_badge_refresh_timer = QTimer()
_badge_refresh_timer.setSingleShot(True)
_badge_refresh_timer.timeout.connect(self._do_badge_refresh)

def _refresh_thumbnail_badges(self):
    # Debounce: restart timer instead of immediate refresh
    self._badge_refresh_timer.stop()
    self._badge_refresh_timer.start(100)  # 100ms debounce
```

**Expected Impact**: 2.07s → ~200ms (single batched update)

---

### 3. Event Filter (1.47s, 146k calls)

**Problem**: `eventFilter` processes every Qt event, even unhandled ones.

**Solution**:
- Early return for unhandled event types
- Cache event type checks
- Reduce function call overhead

**Implementation**:
```python
# In view.py eventFilter
def eventFilter(self, obj, event):
    # Early return for common unhandled events
    event_type = event.type()
    if event_type not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, ...):
        return False  # Let Qt handle it
    
    # ... rest of handler
```

**Expected Impact**: 1.47s → ~500ms

---

### 4. JPEG Loading (7.7s, 30k+ calls)

**Problem**: Loading full JPEG images for thumbnails.

**Solutions**:
1. **Use PIL thumbnail()** - generates smaller thumbnails faster
2. **Lazy loading** - only load visible thumbnails
3. **Virtual scrolling** - only render visible items

**Implementation**:
```python
# In model.py or cache.py
def get_thumbnail_fast(path, size=128):
    """Fast thumbnail using PIL thumbnail() instead of full load."""
    with Image.open(path) as img:
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img.copy()
```

**Expected Impact**: 7.7s → ~1-2s (with virtual scrolling)

---

### 5. I/O Operations (2.11s, 100k file reads)

**Problem**: Too many small file reads.

**Solutions**:
1. **Batch reads** - read multiple files in one operation
2. **Larger read buffers** - read more data per call
3. **Async I/O** - use threading for non-blocking reads

**Implementation**:
- Already using LazyImageLoader (good)
- Consider increasing buffer sizes
- Batch thumbnail cache writes

**Expected Impact**: 2.11s → ~1s

---

### 6. Object Detection (9.2s, 0.51s per call)

**Problem**: Processing images one at a time.

**Solutions**:
1. **Batch detection** - process multiple images per YOLOv8 call
2. **Lazy detection** - only detect when needed (on hover/click)
3. **Skip detection** - for images already processed

**Implementation**:
```python
# In object_detection.py
def detect_objects_batch(image_paths: List[str], config: DetectionConfig):
    """Detect objects in multiple images at once."""
    # YOLOv8 supports batch prediction
    images = [Image.open(path) for path in image_paths]
    results = model.predict(images, batch_size=8)  # Process 8 at once
    return results
```

**Expected Impact**: 9.2s → ~2-3s (with batching)

---

## Cold Cache Optimization Strategy

### Phase 1: Quick Wins (Low effort, high impact)
1. ✅ Debounce badge refresh
2. ✅ Optimize event filter
3. ✅ Pre-load YOLOv8 model on startup

### Phase 2: Medium Effort
4. ⏳ Virtual scrolling (viewport-based rendering)
5. ⏳ Lazy thumbnail loading
6. ⏳ Batch object detection

### Phase 3: Architecture Changes
7. ⏳ Progressive feature extraction
8. ⏳ Incremental prediction
9. ⏳ Chunked image loading

---

## Expected Performance (10K images)

**Current (cold cache, 591 images)**:
- Initial load: ~30s
- Thumbnail generation: ~7.7s
- Object detection: ~9.2s

**Target (with optimizations, 10K images)**:
- Initial load: < 5s (virtual scrolling)
- Thumbnail generation: Progressive, < 1s initial
- Object detection: Lazy, on-demand

