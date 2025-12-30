#!/usr/bin/env python3
"""Analyze profiling results to identify hotspots and memory leaks."""

import pstats
import sys
from pathlib import Path


def analyze_cpu_profile(profile_path: str):
    """Analyze CPU profile and identify hotspots."""
    print("=" * 80)
    print("CPU PROFILE ANALYSIS")
    print("=" * 80)
    
    stats = pstats.Stats(profile_path)
    
    # Sort by cumulative time
    print("\n--- Top 50 functions by cumulative time ---")
    stats.sort_stats('cumulative')
    stats.print_stats(50)
    
    # Sort by total time
    print("\n--- Top 50 functions by total time (self time) ---")
    stats.sort_stats('tottime')
    stats.print_stats(50)
    
    # Sort by call count
    print("\n--- Top 50 functions by call count ---")
    stats.sort_stats('ncalls')
    stats.print_stats(50)
    
    # Find functions with high time per call
    print("\n--- Functions with high time per call (potential bottlenecks) ---")
    stats.sort_stats('tottime')
    all_stats = stats.stats
    
    high_per_call = []
    for func, (cc, nc, tt, ct, callers) in all_stats.items():
        if nc > 0 and tt / nc > 0.01:  # More than 10ms per call
            high_per_call.append((func, tt / nc, tt, nc))
    
    high_per_call.sort(key=lambda x: x[1], reverse=True)
    for func, avg_time, total_time, count in high_per_call[:20]:
        print(f"{func[2]}:{func[1]}:{func[0]} - avg={avg_time*1000:.2f}ms, total={total_time:.2f}s, calls={count}")


def analyze_memory_snapshot(snapshot_path: str):
    """Analyze memory snapshot."""
    print("\n" + "=" * 80)
    print("MEMORY ANALYSIS")
    print("=" * 80)
    
    try:
        import tracemalloc
        snapshot = tracemalloc.Snapshot.load(snapshot_path)
        
        top_stats = snapshot.statistics('lineno')
        total_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
        
        print(f"\nTotal allocated: {total_mb:.2f} MB")
        print(f"Number of allocations: {len(top_stats)}")
        
        print("\n--- Top 50 memory allocations ---")
        for index, stat in enumerate(top_stats[:50], 1):
            size_mb = stat.size / 1024 / 1024
            print(f"{index}. {size_mb:.2f} MB - {stat}")
        
        # Group by filename
        print("\n--- Memory by file ---")
        by_file = {}
        for stat in top_stats:
            filename = stat.traceback[0].filename if stat.traceback else "unknown"
            if filename not in by_file:
                by_file[filename] = 0
            by_file[filename] += stat.size
        
        sorted_files = sorted(by_file.items(), key=lambda x: x[1], reverse=True)
        for filename, size in sorted_files[:20]:
            size_mb = size / 1024 / 1024
            print(f"{size_mb:.2f} MB - {filename}")
            
    except Exception as e:
        print(f"Failed to load memory snapshot: {e}")


def identify_optimizations(profile_path: str):
    """Suggest optimizations based on profile analysis."""
    print("\n" + "=" * 80)
    print("COLD CACHE OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    
    try:
        stats = pstats.Stats(profile_path)
        all_stats = stats.stats
        
        # Find feature extraction hotspots
        feature_hotspots = []
        thumbnail_hotspots = []
        io_hotspots = []
        
        for func, (cc, nc, tt, ct, callers) in all_stats.items():
            func_name = f"{func[2]}:{func[1]}:{func[0]}"
            if 'feature' in func_name.lower() or 'extract' in func_name.lower():
                if tt > 0.1:  # More than 100ms total
                    feature_hotspots.append((func_name, tt, ct, nc))
            elif 'thumbnail' in func_name.lower() or 'pixmap' in func_name.lower():
                if tt > 0.1:
                    thumbnail_hotspots.append((func_name, tt, ct, nc))
            elif 'open' in func_name.lower() or 'read' in func_name.lower() or 'stat' in func_name.lower():
                if tt > 0.05:
                    io_hotspots.append((func_name, tt, ct, nc))
        
        if feature_hotspots:
            print("\n--- Feature Extraction Hotspots ---")
            feature_hotspots.sort(key=lambda x: x[1], reverse=True)
            for func_name, tt, ct, nc in feature_hotspots[:10]:
                print(f"  {func_name}: {tt:.2f}s total, {ct:.2f}s cumulative, {nc} calls")
        
        if thumbnail_hotspots:
            print("\n--- Thumbnail Generation Hotspots ---")
            thumbnail_hotspots.sort(key=lambda x: x[1], reverse=True)
            for func_name, tt, ct, nc in thumbnail_hotspots[:10]:
                print(f"  {func_name}: {tt:.2f}s total, {ct:.2f}s cumulative, {nc} calls")
        
        if io_hotspots:
            print("\n--- I/O Operation Hotspots ---")
            io_hotspots.sort(key=lambda x: x[1], reverse=True)
            for func_name, tt, ct, nc in io_hotspots[:10]:
                print(f"  {func_name}: {tt:.2f}s total, {ct:.2f}s cumulative, {nc} calls")
        
    except Exception as e:
        print(f"Could not analyze for optimizations: {e}")
    
    print("\n--- General Optimization Suggestions ---")
    suggestions = [
        "1. COLD CACHE OPTIMIZATIONS:",
        "   - Batch feature extraction (already done)",
        "   - Parallel thumbnail generation",
        "   - Lazy load images (only load visible thumbnails)",
        "   - Pre-warm cache in background thread",
        "",
        "2. MEMORY OPTIMIZATIONS:",
        "   - Limit pixmap cache size (already done)",
        "   - Clear unused caches periodically",
        "   - Use generators instead of lists for large datasets",
        "   - Release PIL Image objects after conversion",
        "",
        "3. I/O OPTIMIZATIONS:",
        "   - Batch file operations",
        "   - Use async I/O for thumbnail loading",
        "   - Cache file stats (already done)",
        "",
        "4. FOR 10K IMAGES:",
        "   - Virtual scrolling (only render visible thumbnails)",
        "   - Progressive loading (load in chunks)",
        "   - Background feature extraction",
        "   - Incremental prediction (predict as images load)",
    ]
    
    for suggestion in suggestions:
        print(suggestion)


def main():
    # Check for py-spy profile first (modern profiler, all threads/processes)
    pyspy_json = Path("/tmp/app_profile_pyspy.json")
    pyspy_prof = Path("/tmp/app_profile.prof")  # Converted pstats format
    aggregated_path = Path("/tmp/app_profile_aggregated.prof")
    profile_path = Path("/tmp/app_profile.prof")
    memory_path = Path("/tmp/app_memory_final.pkl")
    
    if pyspy_json.exists():
        print("=" * 80)
        print("FOUND PY-SPY PROFILE (ALL THREADS/PROCESSES)")
        print("=" * 80)
        print(f"Profile: {pyspy_json}")
        print("\nTo view py-spy profile, use:")
        print(f"  py-spy top --input {pyspy_json}")
        print(f"  py-spy flamegraph --input {pyspy_json} --output /tmp/flamegraph.svg")
        
        # Try to analyze if converted to pstats format
        if pyspy_prof.exists():
            print("\n" + "=" * 80)
            print("ANALYZING CONVERTED PROFILE (PSTATS FORMAT)")
            print("=" * 80)
            analyze_cpu_profile(str(pyspy_prof))
            identify_optimizations(str(pyspy_prof))
    elif aggregated_path.exists():
        print("=" * 80)
        print("ANALYZING AGGREGATED PROFILE (ALL THREADS)")
        print("=" * 80)
        analyze_cpu_profile(str(aggregated_path))
        identify_optimizations(str(aggregated_path))
        
        # Also show individual thread profiles
        thread_profiles = sorted(Path("/tmp").glob("app_profile_thread_*.prof"))
        if thread_profiles:
            print("\n" + "=" * 80)
            print(f"FOUND {len(thread_profiles)} INDIVIDUAL THREAD PROFILES")
            print("=" * 80)
            for tp in thread_profiles[:5]:  # Show first 5
                print(f"\n--- {tp.name} ---")
                analyze_cpu_profile(str(tp))
    elif profile_path.exists():
        print("=" * 80)
        print("ANALYZING MAIN THREAD PROFILE ONLY")
        print("=" * 80)
        print("(For all threads, use py-spy or aggregated profile)")
        analyze_cpu_profile(str(profile_path))
        identify_optimizations(str(profile_path))
    else:
        print(f"CPU profile not found")
        print("Expected files:")
        print(f"  - {pyspy_json} (py-spy profile - recommended)")
        print(f"  - {aggregated_path} (aggregated cProfile)")
        print(f"  - {profile_path} (main thread cProfile)")
        return 1
    
    # Check for worker process profiles
    worker_profiles = sorted(Path("/tmp").glob("app_profile_worker_*.prof"))
    if worker_profiles:
        print("\n" + "=" * 80)
        print(f"FOUND {len(worker_profiles)} WORKER PROCESS PROFILES")
        print("=" * 80)
        for wp in worker_profiles[:3]:  # Show first 3
            print(f"\n--- {wp.name} ---")
            analyze_cpu_profile(str(wp))
    
    if memory_path.exists():
        analyze_memory_snapshot(str(memory_path))
    else:
        print("\nMemory snapshot not found. Check /tmp/app_memory_final.pkl")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

