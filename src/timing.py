"""Performance timing utilities for profiling operations."""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

# Global timing stats
_timing_stats: dict[str, list[float]] = {}


def reset_stats():
    """Reset all timing statistics."""
    global _timing_stats
    _timing_stats = {}


def get_stats() -> dict[str, dict]:
    """Get timing statistics summary.
    
    Returns:
        Dict mapping operation name to {
            'count': int,
            'total_ms': float,
            'avg_ms': float,
            'min_ms': float,
            'max_ms': float
        }
    """
    global _timing_stats
    result = {}
    for name, times in _timing_stats.items():
        if times:
            total = sum(times) * 1000  # Convert to ms
            result[name] = {
                "count": len(times),
                "total_ms": total,
                "avg_ms": total / len(times),
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
            }
    return result


def log_stats():
    """Log timing statistics summary."""
    stats = get_stats()
    if not stats:
        logging.info("[timing] No timing data collected")
        return
    
    logging.info("[timing] ===== TIMING STATISTICS =====")
    # Sort by total time descending
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total_ms"], reverse=True)
    for name, data in sorted_stats:
        logging.info(
            f"[timing] {name}: count={data['count']}, "
            f"total={data['total_ms']:.1f}ms, "
            f"avg={data['avg_ms']:.2f}ms, "
            f"min={data['min_ms']:.2f}ms, "
            f"max={data['max_ms']:.2f}ms"
        )
    logging.info("[timing] ============================")
    
    # Also write to file for easy access
    try:
        with open("/tmp/photo-derush-timing-stats.txt", "w") as f:
            f.write("===== TIMING STATISTICS =====\n")
            for name, data in sorted_stats:
                f.write(
                    f"{name}: count={data['count']}, "
                    f"total={data['total_ms']:.1f}ms, "
                    f"avg={data['avg_ms']:.2f}ms, "
                    f"min={data['min_ms']:.2f}ms, "
                    f"max={data['max_ms']:.2f}ms\n"
                )
            f.write("============================\n")
        logging.info("[timing] Stats also written to /tmp/photo-derush-timing-stats.txt")
    except Exception as e:
        logging.debug(f"[timing] Failed to write stats file: {e}")


@contextmanager
def time_operation(name: str, log_individual: bool = False):
    """Context manager to time an operation.
    
    Args:
        name: Operation name for statistics
        log_individual: If True, log each individual timing
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        global _timing_stats
        if name not in _timing_stats:
            _timing_stats[name] = []
        _timing_stats[name].append(elapsed)
        if log_individual:
            logging.debug(f"[timing] {name}: {elapsed*1000:.2f}ms")


def timed(name: str, log_individual: bool = False):
    """Decorator to time a function.
    
    Args:
        name: Operation name for statistics (defaults to function name)
        log_individual: If True, log each individual timing
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = name or func.__name__
            with time_operation(op_name, log_individual=log_individual):
                return func(*args, **kwargs)
        return wrapper
    return decorator

