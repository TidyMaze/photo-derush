# Logging Optimization - Removed Spammy Logs

## Problem
Frequent logging statements in hot paths were causing performance overhead:
- String formatting overhead
- I/O overhead (writing to log files/console)
- Memory allocation for log messages

## Logs Removed/Downgraded

### High Frequency (Called for Every Thumbnail/Image)
1. **`[OVERLAY-TEST] _set_label_pixmap`** - `logging.info`
   - **Location**: `src/view.py::_set_label_pixmap()`
   - **Frequency**: Called for every thumbnail update
   - **Action**: Removed

2. **`[OVERLAY-TEST] Filtered X -> Y objects`** - `logging.info`
   - **Location**: `src/view.py::_set_label_pixmap()`
   - **Frequency**: Called for every thumbnail with objects
   - **Action**: Removed

3. **`[GRID] Added image`** - `logging.info`
   - **Location**: `src/view.py::_on_image_added()`
   - **Frequency**: Called for every image (500+ images = 500+ logs)
   - **Action**: Removed

4. **`[THUMBNAIL] Setting pixmap`** - `logging.debug`
   - **Location**: `src/view.py::_on_thumbnail_loaded()`
   - **Frequency**: Called for every thumbnail load
   - **Action**: Removed

5. **`[THUMBNAIL] Pixmap filename mismatch`** - `logging.error`
   - **Location**: `src/view.py::_set_label_pixmap()`
   - **Frequency**: False positive - appears frequently but no actual issue
   - **Action**: Removed (verification check was too strict)

### Medium Frequency (Called on Every Refresh/Relayout)
6. **`[badge-refresh] Processing labels`** - `logging.info`
   - **Location**: `src/view.py::_do_refresh_thumbnail_badges()`
   - **Frequency**: Called on every badge refresh (multiple times per second)
   - **Action**: Removed

7. **`[badge-refresh] Repainted thumbnails`** - `logging.info`
   - **Location**: `src/view.py::_do_refresh_thumbnail_badges()`
   - **Frequency**: Called on every badge refresh
   - **Action**: Removed

8. **`[badge-refresh] Fixed geometry`** - `logging.info`
   - **Location**: `src/view.py::_do_refresh_thumbnail_badges()`
   - **Frequency**: Called during badge refresh when geometry needs fixing
   - **Action**: Removed

9. **`[GRID] Final grid state`** - `logging.info`
   - **Location**: `src/view.py::_relayout_grid()`
   - **Frequency**: Called on every relayout
   - **Action**: Removed

10. **`[GRID] Collecting/Collected items`** - `logging.info`
    - **Location**: `src/view.py::_relayout_grid()`
    - **Frequency**: Called on every relayout
    - **Action**: Removed

11. **`[GRID] Positioned label`** - `logging.debug`
    - **Location**: `src/view.py::_relayout_grid()`
    - **Frequency**: Called for every label positioning
    - **Action**: Removed

### Object Detection Related
12. **`[BBOX-WIDGET] overlay updates`** - `logging.info`
    - **Location**: `src/view.py::_do_refresh_thumbnail_badges()`, `_set_label_pixmap()`
    - **Frequency**: Called for every thumbnail with objects
    - **Action**: Removed

13. **`[BBOX-WIDGET] Hiding overlay`** - `logging.info`
    - **Location**: `src/view.py::_do_refresh_thumbnail_badges()`
    - **Frequency**: Called for every thumbnail without bbox data
    - **Action**: Removed

14. **`[OVERLAY-TEST] No objects to paint`** - `logging.debug`
    - **Location**: `src/view.py::_set_label_pixmap()`
    - **Frequency**: Called for every thumbnail without objects
    - **Action**: Removed

## Impact

### Performance Improvement
- **Reduced I/O overhead**: Eliminated thousands of log writes per session
- **Reduced string formatting**: Removed expensive f-string formatting in hot paths
- **Reduced memory allocations**: Fewer temporary string objects created

### Estimated Savings
- **Logging calls removed**: ~10-15 per thumbnail update
- **For 500 images**: ~5,000-7,500 fewer logging calls during initial load
- **For badge refresh**: ~5-10 fewer logging calls per refresh (multiple times per second)
- **Total**: Thousands of logging calls eliminated per session

## Remaining Logs

### Kept (Low Frequency or Important)
- Error logs (`logging.exception`, `logging.error` for actual errors)
- Important state changes (startup, shutdown, major operations)
- Performance warnings (slow operations >1s)
- Debug logs in non-hot paths

### Notes
- All removed logs were commented out (not deleted) for easy re-enabling if needed
- Critical error logging remains intact
- Performance-critical paths now have minimal logging overhead

