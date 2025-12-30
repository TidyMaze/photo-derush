# Profiling Iteration 4 - Multi-thread/Process Optimization

## Profiling Setup
- **Profiler**: cProfile (main thread) + tracemalloc (memory, all threads)
- **Duration**: 60 seconds
- **Note**: py-spy requires root on macOS, Scalene crashes with Qt, so using cProfile + tracemalloc

## CPU Hotspots Identified

### Top Functions by Cumulative Time
1. **Torch serialization**: 38.9s (model loading)
2. **Sklearn calibration predict_proba**: 19.9s (592 calls) - model inference
3. **EXIF loading**: 28.9s (444 calls) - `model.load_exif()`
4. **PIL Image.open**: 12.8s (1071 calls) - image file I/O
5. **Qt signal emissions**: 7.3s (5130 calls) - `SignalInstance.emit()`
6. **QWidget.show()**: 2.5s (4213 calls) - widget visibility

### Top Functions by Self Time
1. **tracemalloc._group_by**: 4.4s (profiling overhead)
2. **QWidget.show()**: 2.5s (4213 calls)
3. **Other operations**: < 1s each

## Memory Hotspots Identified

### Top Allocations
1. **importlib/bootstrap**: 52.9 MiB (module loading - unavoidable)
2. **importlib/bootstrap**: 10.7 MiB (module loading)
3. **linecache**: 2.2 MiB (source code caching)
4. **features.py**: 1.5 MiB (feature extraction)
5. **model.py**: 861 KiB (EXIF loading)
6. **PIL/TiffImagePlugin**: 634 KiB (image loading)

## Optimizations Applied

### 1. Signal Debouncing
- **Change**: Increased debounce from 200ms to 500ms (5 calls/sec → 2 calls/sec)
- **Impact**: Reduces Qt signal emission overhead
- **File**: `src/viewmodel.py`

### 2. EXIF Loading Batching
- **Change**: Batch EXIF loading in groups of 50 to reduce per-call overhead
- **Impact**: Reduces function call overhead for EXIF collection
- **File**: `src/viewmodel.py`

### 3. Widget Show Filtering
- **Change**: Only call `show()`/`hide()` on widgets that actually need state change
- **Impact**: Reduces redundant Qt widget operations
- **File**: `src/view.py`

## Remaining Hotspots

### High Priority
1. **Qt signal emissions** (7.3s, 5130 calls)
   - **Issue**: Too many signal emissions
   - **Solution**: Further increase debounce, batch multiple state changes
   - **Potential savings**: 3-5s

2. **EXIF loading** (28.9s, 444 calls)
   - **Issue**: Cache exists but still slow (65ms per call average)
   - **Solution**: Pre-warm cache, parallel loading, or reduce EXIF reads
   - **Potential savings**: 20-25s

3. **PIL Image.open** (12.8s, 1071 calls)
   - **Issue**: ~1.8 opens per image (should be ~1)
   - **Solution**: Better caching, reuse Image objects
   - **Potential savings**: 5-7s

### Medium Priority
4. **QWidget.show()** (2.5s, 4213 calls)
   - **Issue**: Still many show() calls despite batching
   - **Solution**: Further reduce redundant visibility changes
   - **Potential savings**: 1-2s

5. **Model inference** (19.9s, 592 calls)
   - **Issue**: Sklearn calibration overhead
   - **Solution**: Batch predictions, optimize model loading
   - **Potential savings**: 5-10s (requires model optimization)

## Next Steps
1. Further optimize signal emissions (batch state changes)
2. Pre-warm EXIF cache on startup
3. Reduce redundant PIL Image.open calls
4. Profile with py-spy (requires sudo) for multi-thread analysis
5. Consider using a different profiler that works with Qt

