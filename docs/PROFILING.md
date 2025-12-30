# Profiling Guide

This app supports **non-intrusive profiling** - no code instrumentation required!

## External Profilers (Recommended)

### py-spy (CPU Profiler - Primary)

**Features:**
- ✅ CPU profiling (all threads/processes/subprocesses)
- ✅ Non-intrusive (attaches to running process)
- ✅ Low overhead (~5%)
- ✅ Thread/process/subprocess tracking
- ❌ Memory profiling (use tracemalloc instead)
- ❌ GPU profiling (not supported)

**Attach to running process:**
```bash
# Find the process ID
PID=$(pgrep -f "python.*app.py" | head -1)

# Record profile (all threads/processes/subprocesses)
py-spy record -o /tmp/app_profile_pyspy.json \
  --pid $PID \
  --subprocesses \
  --threads \
  --rate 100 \
  --duration 60

# Or use poetry run
poetry run py-spy record -o /tmp/app_profile_pyspy.json \
  --pid $PID \
  --subprocesses \
  --threads \
  --rate 100 \
  --duration 60
```

**View results:**
```bash
# Interactive top view
py-spy top --input /tmp/app_profile_pyspy.json

# Generate flamegraph
py-spy flamegraph --input /tmp/app_profile_pyspy.json --output /tmp/flamegraph.svg

# Convert to pstats format (for compatibility)
py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof
python3 tools/analyze_profile.py
```

### Scalene (Modern Profiler)

**Attach to running process:**
```bash
# Find the process ID
PID=$(pgrep -f "python.*app.py" | head -1)

# Profile with HTML output
scalene --profile-all \
  --outfile /tmp/app_profile_scalene.html \
  --pid $PID \
  --duration 60
```

**View results:**
Open `/tmp/app_profile_scalene.html` in a browser.

### One-liner Scripts

**Start app and attach py-spy:**
```bash
# Terminal 1: Start app
PROFILING=1 PHOTO_DERUSH_AUTOQUIT_MS=60000 poetry run python app.py &
APP_PID=$!

# Terminal 2: Attach profiler
sleep 2  # Wait for app to start
poetry run py-spy record -o /tmp/app_profile_pyspy.json \
  --pid $APP_PID \
  --subprocesses \
  --rate 100 \
  --duration 60
```

**Or use a helper script:**
```bash
#!/bin/bash
# profile_app.sh
poetry run python app.py &
APP_PID=$!
echo "App PID: $APP_PID"
sleep 2
poetry run py-spy record -o /tmp/app_profile_pyspy.json \
  --pid $APP_PID \
  --subprocesses \
  --rate 100 \
  --duration 60
```

## Combined Profiling Setup

**CPU Profiling**: Use py-spy externally (attaches to process)
**Memory Profiling**: Use tracemalloc internally (enabled with `PROFILING=1`)

If `PROFILING=1` is set, the app will:
- Enable `tracemalloc` for memory profiling (lightweight, non-intrusive)
- Log the PID for external py-spy attachment
- Dump memory snapshots periodically

**Memory snapshots:**
- `/tmp/app_memory_snapshot.txt` - Human-readable
- `/tmp/app_memory_snapshot.pkl` - Python format
- `/tmp/app_memory_final.txt` - Final snapshot on exit

**Complete profiling workflow:**
1. Start app with `PROFILING=1` (enables memory profiling)
2. Attach py-spy for CPU profiling: `py-spy record --pid <PID> --subprocesses --threads`
3. Analyze both CPU (py-spy) and memory (tracemalloc) results

## Advantages of External Profiling

1. **No code changes** - Profiler attaches to running process
2. **All threads/processes** - Automatically profiles everything
3. **Lower overhead** - Sampling-based, minimal impact
4. **Start/stop anytime** - Attach/detach without restarting app
5. **Multiple profilers** - Can use different profilers for different runs

## Profiler Comparison

| Profiler | Type | Threads | Processes | Overhead | Output | CPU | Memory | GPU |
|----------|------|---------|----------|----------|--------|-----|--------|-----|
| py-spy | Sampling | ✅ | ✅ | Low (~5%) | JSON, SVG | ✅ | ❌ | ❌ |
| scalene | Sampling | ✅ | ✅ | Low (10-20%) | HTML | ✅ Line-level | ✅ Line-level | ✅ |
| cProfile | Tracing | ❌ | ❌ | High (50-100%) | pstats | ✅ Function-level | ❌ | ❌ |
| tracemalloc | Memory | ✅ | ❌ | Low (~5%) | Text | ❌ | ✅ Allocation-level | ❌ |

**See [PROFILER_COMPARISON.md](PROFILER_COMPARISON.md) for detailed comparison.**

## Tips

- **For CPU profiling**: Use **py-spy** (primary, lowest overhead, threads/processes/subprocesses)
- **For memory profiling**: Use **tracemalloc** (enabled with `PROFILING=1`, lightweight) or scalene (detailed)
- **For GPU profiling**: Use scalene or specialized GPU profilers (NVIDIA Nsight, PyTorch Profiler)
- **For quick analysis**: Use py-spy's `top` command
- **For detailed reports**: Generate flamegraphs (py-spy) or HTML reports (scalene)
- **For compatibility**: Convert py-spy output to pstats format
- **For production**: Use py-spy (lowest overhead, safest)

**Note**: py-spy is CPU-only. For memory profiling, tracemalloc is already integrated (enabled with `PROFILING=1`).

**See [PROFILER_CHOICE.md](PROFILER_CHOICE.md) for detailed comparison and recommendations.**

