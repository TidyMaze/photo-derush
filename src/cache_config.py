"""Cache configuration and disable functionality.

Provides a centralized way to disable all caches via environment variable.
When disabled, all cache lookups return MISS (None/empty) to force recomputation.
"""

import os
import logging

# Environment variable to disable all caches
DISABLE_CACHE_ENV_VAR = "PHOTO_DERUSH_DISABLE_CACHE"

# Cached value (computed once at import time for performance)
_CACHE_DISABLED: bool | None = None


def is_cache_disabled() -> bool:
    """Check if caching is disabled via environment variable.
    
    Returns:
        True if PHOTO_DERUSH_DISABLE_CACHE is set to a truthy value, False otherwise.
        Result is cached after first call for performance.
    """
    global _CACHE_DISABLED
    if _CACHE_DISABLED is None:
        env_value = os.environ.get(DISABLE_CACHE_ENV_VAR, "").lower()
        _CACHE_DISABLED = env_value in ("1", "true", "yes", "on")
        if _CACHE_DISABLED:
            logging.info(f"[cache_config] Caching disabled via {DISABLE_CACHE_ENV_VAR}")
    return _CACHE_DISABLED

