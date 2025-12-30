# Profiler Choice: Scalene vs Memray vs py-spy

## Decision: **py-spy** (Primary) + **Scalene** (Secondary)

## Quick Comparison

| Feature | py-spy | Scalene | Memray |
|---------|--------|---------|--------|
| **CPU Profiling** | ✅ Function-level | ✅ Line-level | ❌ |
| **Memory Profiling** | ❌ | ✅ Line-level | ✅ Deep analysis |
| **GPU Profiling** | ❌ | ✅ | ❌ |
| **Code Changes** | ❌ None | ❌ None | ✅ Required |
| **Overhead** | ~5% | 10-20% | 5-15% |
| **Threads/Processes** | ✅ All | ✅ All | ✅ All |
| **Output** | JSON, SVG | HTML | HTML, JSON |
| **Production Safe** | ✅ Yes | ⚠️ Caution | ⚠️ Caution |
| **Platform** | All | All | Linux/macOS only |
| **Memory Leak Detection** | ❌ | ✅ Good | ✅ Excellent |
| **Flamegraphs** | ✅ Excellent | ✅ | ✅ Excellent |

## Why py-spy (Primary Choice)

### ✅ Matches Project Needs

1. **CPU profiling is critical** for this project:
   - Duplicate grouping (17.4s bottleneck)
   - Object detection (9.2s)
   - Event filtering (1.47s)
   - JPEG loading (7.7s)

2. **Lowest overhead** (~5%):
   - Safe for production profiling
   - Can profile long-running sessions
   - Minimal impact on real performance

3. **Non-intrusive**:
   - Already set up in project
   - Attach to running process
   - No code changes needed

4. **Scriptable output**:
   - JSON format for automated analysis
   - Works with existing `tools/analyze_profile.py`
   - Flamegraphs for visualization

5. **Mature & reliable**:
   - Well-established tool
   - Handles multiprocessing workers
   - Cross-platform

### ❌ Limitations
- Function-level only (can't see line-level hotspots)
- CPU only (no memory profiling)

## Why Scalene (Secondary Choice)

### ✅ Use When You Need:

1. **Comprehensive profiling**:
   - CPU + Memory + GPU in one tool
   - Line-level precision
   - Python vs native code separation

2. **Memory analysis**:
   - Better than tracemalloc for detailed memory profiling
   - Line-level allocation tracking
   - Beautiful HTML reports

3. **AI suggestions**:
   - Automatic optimization recommendations
   - Helps identify bottlenecks

### ❌ Limitations
- Higher overhead (10-20%)
- HTML-only output (less scriptable)
- Newer tool (less mature)

## Why NOT Memray

### ❌ Doesn't Match Project Needs

1. **Requires code changes**:
   - Must instrument code with decorators/context managers
   - Breaks non-intrusive profiling approach
   - More invasive than py-spy/Scalene

2. **Memory-only**:
   - No CPU profiling
   - Project needs CPU profiling more (CPU bottlenecks are bigger)
   - tracemalloc already covers basic memory profiling

3. **Platform limitation**:
   - Linux/macOS only (project may need Windows support)

4. **Redundant**:
   - tracemalloc already integrated for memory
   - Scalene does memory + CPU together

### ✅ Memray Would Be Good If:
- You ONLY needed deep memory profiling
- You could accept code instrumentation
- You were on Linux/macOS only
- You needed the best memory leak detection

## Recommendation for This Project

### Primary: **py-spy**
```bash
# CPU profiling (production-safe, low overhead)
py-spy record -o /tmp/profile.json \
  --pid $(pgrep -f "python.*app.py") \
  --subprocesses \
  --rate 100 \
  --duration 60
```

**Use for:**
- Regular CPU profiling
- Production profiling
- Quick hotspot identification
- Automated analysis workflows

### Secondary: **Scalene**
```bash
# Comprehensive profiling (CPU + Memory + GPU)
scalene --profile-all \
  --outfile /tmp/profile.html \
  --pid $(pgrep -f "python.*app.py") \
  --duration 60
```

**Use for:**
- Deep analysis sessions
- Memory leak investigation
- Line-level optimization
- When you need AI suggestions

### Skip: **Memray**
- Doesn't fit non-intrusive approach
- Memory-only (CPU is bigger bottleneck)
- tracemalloc already covers memory needs

## Implementation

### Update `scripts/profile_app.sh`
```bash
# Already supports py-spy and scalene
./scripts/profile_app.sh py-spy 60    # CPU profiling
./scripts/profile_app.sh scalene 60  # Comprehensive
```

### Update `docs/PROFILING.md`
- Keep py-spy as primary recommendation
- Add Scalene as secondary option
- Remove/update Memray references if any

## Summary

**Choose py-spy** because:
1. ✅ CPU profiling is the main need (biggest bottlenecks)
2. ✅ Lowest overhead (production-safe)
3. ✅ Non-intrusive (matches current setup)
4. ✅ Scriptable output (works with existing tools)
5. ✅ Mature and reliable

**Add Scalene** for:
- Comprehensive analysis when needed
- Memory profiling (better than tracemalloc)
- Line-level insights

**Skip Memray** because:
- Requires code changes (breaks non-intrusive approach)
- Memory-only (CPU is bigger bottleneck)
- tracemalloc already covers memory needs

