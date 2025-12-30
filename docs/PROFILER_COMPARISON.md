# Profiler Comparison: Scalene vs Others

## Quick Summary

| Feature | Scalene | py-spy | cProfile | tracemalloc |
|---------|---------|--------|----------|-------------|
| **CPU Profiling** | ✅ Line-level | ✅ Function-level | ✅ Function-level | ❌ |
| **Memory Profiling** | ✅ Line-level | ❌ | ❌ | ✅ Allocation-level |
| **GPU Profiling** | ✅ | ❌ | ❌ | ❌ |
| **Python vs Native** | ✅ Separates | ❌ | ❌ | ❌ |
| **Code Changes** | ❌ None | ❌ None | ✅ Required | ✅ Required |
| **Threads** | ✅ All | ✅ All | ❌ Main only | ✅ All |
| **Processes** | ✅ All | ✅ All | ❌ Main only | ❌ Main only |
| **Overhead** | Low (10-20%) | Low (~5%) | High (50-100%) | Low (~5%) |
| **Output Format** | HTML | JSON, SVG | pstats | Text |
| **AI Suggestions** | ✅ | ❌ | ❌ | ❌ |
| **Line-level Detail** | ✅ | ❌ | ❌ | ❌ |

## Detailed Comparison

### Scalene

**Strengths:**
- **Comprehensive**: Profiles CPU, memory, AND GPU in one tool
- **Line-level precision**: Shows exactly which lines consume resources
- **Python vs Native separation**: Distinguishes Python code from C/C++ libraries
- **AI-powered insights**: Suggests optimizations automatically
- **Beautiful HTML reports**: Interactive, color-coded, easy to navigate
- **Zero code changes**: Attach to running process
- **Low overhead**: 10-20% performance impact (sampling-based)

**Weaknesses:**
- **Larger install**: More dependencies than py-spy
- **HTML-only output**: Less scriptable than JSON
- **Newer tool**: Less mature ecosystem than py-spy

**Best for:**
- Finding memory leaks (line-level allocation tracking)
- GPU-accelerated code (PyTorch, TensorFlow)
- Detailed optimization analysis
- When you need AI suggestions

**Example:**
```bash
scalene --profile-all \
  --outfile /tmp/profile.html \
  --pid $(pgrep -f "python.*app.py") \
  --duration 60
```

### py-spy

**Strengths:**
- **Lightweight**: Minimal dependencies, fast install
- **Multiple formats**: JSON, SVG flamegraphs, pstats conversion
- **Scriptable**: JSON output easy to parse/analyze
- **Mature**: Well-established, widely used
- **Very low overhead**: ~5% performance impact
- **Flamegraphs**: Excellent visualization

**Weaknesses:**
- **Function-level only**: Can't see line-level hotspots
- **CPU only**: No memory or GPU profiling
- **No AI suggestions**: Manual analysis required

**Best for:**
- Quick CPU hotspot identification
- Production profiling (lowest overhead)
- Scripted analysis workflows
- Flamegraph visualization

**Example:**
```bash
py-spy record -o /tmp/profile.json \
  --pid $(pgrep -f "python.*app.py") \
  --subprocesses \
  --rate 100 \
  --duration 60
```

### cProfile

**Strengths:**
- **Built-in**: No installation needed
- **Detailed call counts**: Exact function call statistics
- **Standard format**: pstats format widely supported

**Weaknesses:**
- **High overhead**: 50-100% performance impact
- **Code changes required**: Must instrument code
- **Main thread only**: Doesn't profile threads/processes
- **Function-level only**: No line-level detail
- **CPU only**: No memory profiling

**Best for:**
- Quick function-level analysis
- When you can't install external tools
- Simple scripts (not production)

**Example:**
```python
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# ... your code ...
profiler.disable()
profiler.dump_stats("/tmp/profile.prof")
```

### tracemalloc

**Strengths:**
- **Built-in**: No installation needed
- **Memory-focused**: Tracks allocations precisely
- **Low overhead**: ~5% performance impact
- **All threads**: Profiles all threads in process

**Weaknesses:**
- **Memory only**: No CPU profiling
- **Code changes required**: Must enable in code
- **Allocation-level**: Not line-level (shows where memory was allocated)
- **Text output**: Less visual than Scalene

**Best for:**
- Memory leak detection
- Allocation tracking
- When you need built-in tools only

**Example:**
```python
import tracemalloc
tracemalloc.start()
# ... your code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
```

## Recommendations for This Project

### For CPU Profiling
1. **py-spy** - Best for quick CPU hotspot identification, lowest overhead
2. **Scalene** - Best for detailed line-level CPU analysis

### For Memory Profiling
1. **Scalene** - Best overall (line-level, beautiful reports)
2. **tracemalloc** - Good fallback (built-in, lightweight)

### For Comprehensive Analysis
1. **Scalene** - Only profiler that does CPU + Memory + GPU in one tool

### For Production Profiling
1. **py-spy** - Lowest overhead, safest for production
2. **Scalene** - Good if you need memory insights too

## When to Use Each

### Use Scalene when:
- ✅ You need memory profiling (leaks, allocations)
- ✅ You want line-level precision
- ✅ You're using GPU-accelerated code
- ✅ You want AI-powered suggestions
- ✅ You need comprehensive profiling in one tool

### Use py-spy when:
- ✅ You only need CPU profiling
- ✅ You want lowest overhead
- ✅ You need scriptable JSON output
- ✅ You want flamegraphs
- ✅ You're profiling production systems

### Use cProfile when:
- ✅ You can't install external tools
- ✅ You need exact call counts
- ✅ You're profiling simple scripts

### Use tracemalloc when:
- ✅ You only need memory profiling
- ✅ You can't install external tools
- ✅ You need built-in solution

## Installation

```bash
# Scalene
pip install scalene
# or
poetry add scalene

# py-spy
pip install py-spy
# or
poetry add py-spy

# cProfile & tracemalloc
# Built-in to Python, no installation needed
```

## Output Examples

### Scalene HTML Report
- Color-coded lines (red = CPU hotspot, yellow = memory hotspot)
- Interactive navigation
- Line-by-line statistics
- AI suggestions panel

### py-spy JSON
```json
{
  "process_name": "python",
  "samples": [...],
  "functions": [...]
}
```

### py-spy Flamegraph
- Visual call stack representation
- Width = time spent
- Height = call depth

## Performance Impact

| Profiler | Overhead | Production Safe? |
|----------|----------|------------------|
| Scalene | 10-20% | ⚠️ Use with caution |
| py-spy | ~5% | ✅ Yes |
| cProfile | 50-100% | ❌ No |
| tracemalloc | ~5% | ✅ Yes |

## Conclusion

**For this photo-derush project:**
- **Primary**: Use **py-spy** for CPU profiling (lowest overhead, scriptable)
- **Secondary**: Use **Scalene** for comprehensive analysis (CPU + memory + GPU)
- **Memory-only**: Use **tracemalloc** (already integrated, lightweight)

Scalene is superior for **comprehensive analysis** but py-spy is better for **production CPU profiling** due to lower overhead.

