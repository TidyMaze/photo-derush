# py-spy Setup for CPU Profiling

## Overview

**py-spy** is the primary CPU profiler for this project. It profiles:
- ✅ All threads
- ✅ All processes
- ✅ All subprocesses
- ✅ Low overhead (~5%)
- ✅ Non-intrusive (no code changes)

**Note**: py-spy is **CPU-only**. For memory profiling, use tracemalloc (enabled with `PROFILING=1`).

## Installation

```bash
# Using pip
pip install py-spy

# Using poetry
poetry add py-spy
```

## Basic Usage

### 1. Attach to Running Process

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
```

### 2. Using Helper Script

```bash
# Start app and profile automatically
./scripts/profile_app.sh py-spy 60
```

### 3. Live Monitoring

```bash
# Interactive top view (live updates)
py-spy top --pid $PID --subprocesses --threads
```

## Command Options

### Essential Flags

- `--pid <PID>`: Process ID to profile
- `--subprocesses`: Include all Python subprocesses
- `--threads`: Profile all threads (default, but explicit is better)
- `--rate <N>`: Sampling rate in Hz (default: 100, higher = more overhead)
- `--duration <N>`: Duration in seconds
- `-o <file>`: Output file

### Output Formats

- `-o profile.json`: JSON format (scriptable)
- `-o profile.svg`: SVG flamegraph (visual)
- `-o profile.prof`: pstats format (compatible with existing tools)

## Viewing Results

### 1. Interactive Top View

```bash
py-spy top --input /tmp/app_profile_pyspy.json
```

Shows top functions by CPU time, updating interactively.

### 2. Flamegraph

```bash
py-spy flamegraph --input /tmp/app_profile_pyspy.json --output /tmp/flamegraph.svg
```

Generates an interactive SVG flamegraph. Open in browser.

### 3. Convert to pstats

```bash
py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof
python3 tools/analyze_profile.py
```

Converts to pstats format for compatibility with existing analysis tools.

## Profiling Threads/Processes/Subprocesses

### Threads

py-spy automatically profiles all threads. Use `--threads` flag explicitly:

```bash
py-spy record --pid $PID --threads -o profile.json
```

### Processes

For multiprocessing workers, use `--subprocesses`:

```bash
py-spy record --pid $PID --subprocesses -o profile.json
```

This includes:
- Main process
- All Python subprocesses spawned by multiprocessing
- Worker processes from ThreadPoolExecutor/ProcessPoolExecutor

### Complete Command

```bash
py-spy record \
  --pid $PID \
  --subprocesses \
  --threads \
  --rate 100 \
  --duration 60 \
  -o /tmp/profile.json
```

## Memory Profiling (Separate)

py-spy does **not** profile memory. Use tracemalloc instead:

```bash
# Start app with memory profiling
PROFILING=1 poetry run python app.py

# Memory snapshots are automatically saved to:
# - /tmp/app_memory_snapshot.txt
# - /tmp/app_memory_final.txt
```

## GPU Profiling (Not Supported)

py-spy does **not** profile GPU. For GPU profiling, use:
- **Scalene**: `scalene --profile-all --pid $PID`
- **NVIDIA Nsight**: For CUDA applications
- **PyTorch Profiler**: For PyTorch models

## Complete Workflow

### 1. Start App with Memory Profiling

```bash
PROFILING=1 poetry run python app.py &
APP_PID=$!
```

### 2. Attach py-spy for CPU Profiling

```bash
py-spy record \
  --pid $APP_PID \
  --subprocesses \
  --threads \
  --rate 100 \
  --duration 60 \
  -o /tmp/cpu_profile.json
```

### 3. Analyze Results

```bash
# CPU analysis
py-spy top --input /tmp/cpu_profile.json
py-spy flamegraph --input /tmp/cpu_profile.json --output /tmp/cpu_flamegraph.svg

# Memory analysis (from tracemalloc)
cat /tmp/app_memory_final.txt
```

## Troubleshooting

### Permission Denied

On Linux, you may need `sudo`:

```bash
sudo py-spy record --pid $PID --subprocesses --threads -o profile.json
```

### Process Not Found

Make sure the process is running:

```bash
ps aux | grep "python.*app.py"
```

### No Subprocesses Detected

Ensure `--subprocesses` flag is used:

```bash
py-spy record --pid $PID --subprocesses --threads -o profile.json
```

## Performance Impact

- **Overhead**: ~5% CPU
- **Memory**: Minimal (~10-20 MB)
- **Production Safe**: Yes (low overhead)

## Best Practices

1. **Use appropriate sampling rate**:
   - Development: 100-200 Hz (more detail)
   - Production: 50-100 Hz (lower overhead)

2. **Profile for sufficient duration**:
   - Short operations: 10-30 seconds
   - Long operations: 60-300 seconds

3. **Combine with memory profiling**:
   - Use tracemalloc for memory (already integrated)
   - Use py-spy for CPU (external attachment)

4. **Profile all components**:
   - Always use `--subprocesses` for multiprocessing
   - Always use `--threads` for threading

## Example Output

```json
{
  "process_name": "python",
  "samples": [
    {
      "thread_id": 12345,
      "stack": [
        "main",
        "app.main",
        "viewmodel.load_images",
        "features.batch_extract_features"
      ],
      "timestamp": 1234567890.123
    }
  ],
  "functions": {
    "features.batch_extract_features": {
      "samples": 1523,
      "time": 12.45
    }
  }
}
```

## Integration with Existing Tools

The project's `tools/analyze_profile.py` can process py-spy output:

```bash
# Convert to pstats format
py-spy convert -i /tmp/app_profile_pyspy.json -o /tmp/app_profile.prof

# Analyze
python3 tools/analyze_profile.py
```

