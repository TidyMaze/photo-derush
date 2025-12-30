"""Utilities for multi-threaded and multi-process profiling."""

import cProfile
import logging
import os
import threading
from pathlib import Path
from typing import Any


# Global profiler instances per thread
_thread_profilers: dict[int, cProfile.Profile] = {}
_thread_profilers_lock = threading.Lock()


def _thread_profile_func(frame, event, arg):
    """Profile function for threading.setprofile()."""
    thread_id = threading.get_ident()
    with _thread_profilers_lock:
        profiler = _thread_profilers.get(thread_id)
        if profiler:
            # Use the profiler's internal dispatch
            try:
                return profiler.trace_dispatch(frame, event, arg)
            except ValueError:
                # Another profiler is already active (e.g., main thread profiler)
                # This can happen if threading.setprofile conflicts with cProfile.Profile.enable()
                # Just return None to skip profiling this thread
                return None
        else:
            # Create profiler for this thread on first call
            try:
                profiler = cProfile.Profile()
                profiler.enable()
                _thread_profilers[thread_id] = profiler
                logging.debug(f"[PROFILING] Created profiler for thread {thread_id}")
                return profiler.trace_dispatch(frame, event, arg)
            except ValueError:
                # Another profiler is already active - skip this thread
                logging.debug(f"[PROFILING] Skipping thread {thread_id} (profiler conflict)")
                return None
    return None


def enable_thread_profiling():
    """Enable profiling for all threads."""
    if os.environ.get("PROFILING") != "1":
        return
    
    # Set profile function for all threads
    threading.setprofile(_thread_profile_func)
    logging.info("[PROFILING] Enabled thread profiling via threading.setprofile()")
    
    # Create profiler for main thread
    thread_id = threading.get_ident()
    with _thread_profilers_lock:
        _thread_profilers[thread_id] = cProfile.Profile()
        _thread_profilers[thread_id].enable()
        logging.info(f"[PROFILING] Created profiler for main thread {thread_id}")


def create_thread_profiler(thread_id: int | None = None) -> cProfile.Profile:
    """Create a profiler for a specific thread."""
    if thread_id is None:
        thread_id = threading.get_ident()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    with _thread_profilers_lock:
        _thread_profilers[thread_id] = profiler
    
    logging.debug(f"[PROFILING] Created profiler for thread {thread_id}")
    return profiler


def get_all_thread_profilers() -> dict[int, cProfile.Profile]:
    """Get all thread profilers."""
    with _thread_profilers_lock:
        return _thread_profilers.copy()


def dump_all_profiles(output_dir: str = "/tmp"):
    """Dump all thread profiles to files."""
    if os.environ.get("PROFILING") != "1":
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with _thread_profilers_lock:
        for thread_id, profiler in _thread_profilers.items():
            try:
                profiler.disable()
                profile_file = output_path / f"app_profile_thread_{thread_id}.prof"
                profiler.dump_stats(str(profile_file))
                logging.info(f"[PROFILING] Dumped thread {thread_id} profile to {profile_file}")
            except Exception as e:
                logging.warning(f"[PROFILING] Failed to dump thread {thread_id} profile: {e}")


def aggregate_profiles(output_dir: str = "/tmp") -> cProfile.Profile | None:
    """Aggregate all thread profiles into a single profile."""
    if os.environ.get("PROFILING") != "1":
        return None
    
    import pstats
    from io import StringIO
    
    # Use pstats.Stats to merge profiles (cProfile.Profile doesn't have add method)
    aggregated_stats = None
    
    with _thread_profilers_lock:
        for thread_id, profiler in _thread_profilers.items():
            try:
                profiler.disable()
                # Dump individual profile first, then add to aggregated
                temp_file = Path(output_dir) / f"app_profile_thread_{thread_id}_temp.prof"
                profiler.dump_stats(str(temp_file))
                
                # Create Stats from dumped file
                stats = pstats.Stats(str(temp_file))
                
                if aggregated_stats is None:
                    aggregated_stats = stats
                else:
                    aggregated_stats.add(stats)
                logging.debug(f"[PROFILING] Aggregated thread {thread_id} profile")
            except Exception as e:
                logging.warning(f"[PROFILING] Failed to aggregate thread {thread_id} profile: {e}")
    
    if aggregated_stats is None:
        logging.warning("[PROFILING] No thread profiles to aggregate")
        return None
    
    # Save aggregated profile
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregated_file = output_path / "app_profile_aggregated.prof"
    aggregated_stats.dump_stats(str(aggregated_file))
    logging.info(f"[PROFILING] Aggregated profile saved to {aggregated_file}")
    
    # Create a Profile object for return (for compatibility)
    aggregated = cProfile.Profile()
    aggregated.load_stats(str(aggregated_file))
    return aggregated


# Multiprocessing worker profiling
def init_worker_profiler():
    """Initialize profiler in a multiprocessing worker."""
    if os.environ.get("PROFILING") != "1":
        return None
    
    import os
    worker_id = os.getpid()
    profiler = cProfile.Profile()
    profiler.enable()
    logging.info(f"[PROFILING] Enabled profiler for worker process {worker_id}")
    return profiler


def dump_worker_profile(profiler: cProfile.Profile | None, worker_id: int | None = None, output_dir: str = "/tmp"):
    """Dump a worker process profile."""
    if profiler is None or os.environ.get("PROFILING") != "1":
        return
    
    if worker_id is None:
        import os
        worker_id = os.getpid()
    
    try:
        profiler.disable()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        profile_file = output_path / f"app_profile_worker_{worker_id}.prof"
        profiler.dump_stats(str(profile_file))
        logging.info(f"[PROFILING] Dumped worker {worker_id} profile to {profile_file}")
    except Exception as e:
        logging.warning(f"[PROFILING] Failed to dump worker {worker_id} profile: {e}")

