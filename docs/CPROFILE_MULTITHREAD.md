# cProfile Multi-Thread/Process Profiling

## Limitations

**cProfile by default**: Only profiles the **main thread**. It does NOT automatically profile:
- Other threads (worker threads, Qt threads, etc.)
- Subprocesses (multiprocessing workers)

## Multi-Thread Support

cProfile **can** profile multiple threads, but requires manual setup using `threading.setprofile()`.

### Current Implementation

We have `src/profiling_utils.py` that implements multi-thread profiling:

```python
def enable_thread_profiling():
    """Enable profiling for all threads."""
    threading.setprofile(_thread_profile_func)
    # Creates a cProfile.Profile() instance for each thread
```

### How It Works

1. **`threading.setprofile()`**: Sets a profile function that's called for each thread
2. **Per-thread profilers**: Each thread gets its own `cProfile.Profile()` instance
3. **Aggregation**: Profiles from all threads can be aggregated into one

### Current Status

**NOT ENABLED** in `app.py`. Currently only main thread is profiled:

```python
# app.py (current)
cpu_profiler = cProfile.Profile()
cpu_profiler.enable()  # Only main thread
```

### To Enable Multi-Thread Profiling

Add to `app.py`:

```python
from src.profiling_utils import enable_thread_profiling, aggregate_profiles

if os.environ.get("PROFILING") == "1":
    enable_thread_profiling()  # Enable for all threads
    # ... rest of code ...
    
    # On exit, aggregate all thread profiles
    aggregate_profiles("/tmp")
```

## Multi-Process Support

cProfile **can** profile multiple processes, but requires:
1. **Separate profiler per process**: Each worker process needs its own `cProfile.Profile()`
2. **Manual aggregation**: Collect profiles from all processes and merge them

### Current Implementation

We have `src/profiling_utils.py` with `init_worker_profiler()`:

```python
def init_worker_profiler():
    """Initialize profiler in a multiprocessing worker."""
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler
```

### Usage in Multiprocessing

```python
# In worker process initialization
from src.profiling_utils import init_worker_profiler

def worker_init():
    profiler = init_worker_profiler()
    # ... worker code ...
    dump_worker_profile(profiler)
```

## Comparison with py-spy

| Feature | cProfile | py-spy |
|---------|----------|--------|
| **Main thread** | ✅ Native | ✅ Native |
| **Other threads** | ⚠️ Manual setup (`threading.setprofile()`) | ✅ Automatic (`--threads`) |
| **Subprocesses** | ⚠️ Manual setup (per-process profiler) | ✅ Automatic (`--subprocesses`) |
| **Overhead** | High (instrumentation) | Low (sampling) |
| **Root required** | ❌ No | ⚠️ Yes (on macOS) |
| **Ease of use** | ⚠️ Complex | ✅ Simple |

## Recommendation

For **comprehensive multi-thread/process profiling**:
- **Use py-spy** (if sudo is available): Automatic, low overhead, covers all threads/processes
- **Use cProfile multi-thread** (if no sudo): Manual setup, higher overhead, but works

## Current Setup

We're using **cProfile (main thread only)** because:
1. No sudo required
2. Simple setup
3. Good enough for identifying main-thread hotspots

To get **all threads/processes**, we'd need to:
1. Enable `enable_thread_profiling()` in `app.py`
2. Add worker profiling to multiprocessing code
3. Aggregate profiles on exit

Or use **py-spy** with sudo for automatic coverage.

