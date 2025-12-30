# Profiling Iteration 3 - Current Hotspots

## Profile Summary
- **Total time**: 34.567s
- **Function calls**: 18,132,321
- **Profiling duration**: 60 seconds

## Top Hotspots (Self Time)

1. **tracemalloc._group_by**: 4.365s (profiling overhead - ignore)
2. **QWidget.show()**: 2.489s (2021 calls, 0.001s per call)
3. **imagehash.__sub__**: 0.836s (174770 calls) - O(n²) comparison, already optimized
4. **TiffImagePlugin.load**: 0.882s (2401 calls) - EXIF loading overhead

## Top Hotspots (Cumulative Time)

1. **predict_proba (sklearn)**: 13.287s - ML inference (unavoidable)
2. **load_exif()**: 12.927s (2958 calls, 1226 cache misses)
3. **Image.open()**: 10.680s (1226 calls) - EXIF extraction
4. **Signal emits**: 8.386s (5190 calls, 0.002s per call)
5. **QWidget.show()**: 3.067s (2021 calls) - already batched

## Optimization Opportunities

### 1. Signal Emissions (8.4s, 5190 calls)
- **Impact**: High (8.4s total)
- **Difficulty**: Medium (requires architectural changes)
- **Approach**: Batch or debounce signal emissions
- **Files**: `src/services.py`, `src/taskrunner.py`, `src/viewmodel.py`

### 2. EXIF Loading (12.9s, 1226 cache misses)
- **Impact**: High (12.9s total)
- **Difficulty**: Hard (requires alternative EXIF library or pre-warming)
- **Approach**: 
  - Pre-warm EXIF cache in background thread
  - Use faster EXIF library (piexif, exifread)
  - Batch EXIF extraction
- **Files**: `src/model.py`

### 3. QWidget.show() (2.5s, 2021 calls)
- **Impact**: Medium (2.5s total)
- **Difficulty**: Low (already batched, could reduce further)
- **Approach**: Further reduce show() calls by checking visibility before showing
- **Files**: `src/view.py`

### 4. Image Hash Comparisons (0.8s, 174770 calls)
- **Impact**: Low (0.8s total)
- **Difficulty**: Low (already optimized)
- **Status**: Already optimized in previous iteration

## Next Steps

1. **Signal emissions**: Investigate batching or debouncing opportunities
2. **EXIF pre-warming**: Load EXIF data in background thread during app startup
3. **Widget visibility**: Add visibility checks before show() calls

## Notes

- Context manager optimization for EXIF loading didn't help (PIL already handles file closing)
- Most time is spent in unavoidable operations (ML inference, file I/O)
- Signal emissions are the biggest remaining optimization target

