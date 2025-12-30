"""Disk cache for perceptual hashes to avoid recomputing them."""

import hashlib
import json
import logging
import os

from .cache import CACHE_DIR


def _phash_cache_key(path: str) -> str:
    """Generate cache key for a file based on path, mtime, and size."""
    try:
        stat = os.stat(path)
        key = f"{path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.sha256(key.encode()).hexdigest()
    except Exception:
        # On error, use path only
        return hashlib.sha256(path.encode()).hexdigest()


class PerceptualHashCache:
    """Disk cache for perceptual hashes."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        # CACHE_DIR is already expanded, but expand again if custom path provided
        self.cache_dir = os.path.expanduser(cache_dir) if cache_dir != CACHE_DIR else cache_dir
        self.cache_file = os.path.join(self.cache_dir, "perceptual_hashes.json")
        self._cache: dict[str, str] = {}  # cache_key -> hash_string
        self._loaded = False
        logging.info(f"[phash_cache] Initialized with cache_file: {self.cache_file} (exists: {os.path.exists(self.cache_file)})")

    def _load_cache(self):
        """Load cache from disk."""
        if self._loaded:
            logging.debug(f"[phash_cache] Cache already loaded ({len(self._cache)} entries)")
            return
        self._loaded = True
        cache_file_path = self.cache_file
        if not os.path.exists(cache_file_path):
            logging.debug(f"[phash_cache] Cache file does not exist: {cache_file_path}")
            return
        try:
            file_size = os.path.getsize(cache_file_path)
            logging.debug(f"[phash_cache] Cache file size: {file_size} bytes")
            with open(cache_file_path, "r") as f:
                loaded = json.load(f)
                logging.debug(f"[phash_cache] JSON parsed, type: {type(loaded)}, len: {len(loaded) if isinstance(loaded, dict) else 'N/A'}")
                if isinstance(loaded, dict):
                    self._cache = loaded
                    logging.info(f"[phash_cache] Loaded {len(self._cache)} cached hashes from {cache_file_path}")
                else:
                    logging.warning(f"[phash_cache] Cache file has invalid format (expected dict, got {type(loaded)})")
                    self._cache = {}
        except json.JSONDecodeError as e:
            logging.warning(f"[phash_cache] Cache file is corrupted (JSON error at line {e.lineno}): {e}", exc_info=True)
            self._cache = {}
        except Exception as e:
            logging.warning(f"[phash_cache] Failed to load cache from {cache_file_path}: {e}", exc_info=True)
            self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file_path = os.path.expanduser(self.cache_file) if self.cache_file.startswith("~") else self.cache_file
            # Don't overwrite existing cache with empty dict
            if len(self._cache) == 0 and os.path.exists(cache_file_path):
                file_size = os.path.getsize(cache_file_path)
                if file_size > 0:
                    logging.warning(f"[phash_cache] Skipping save: cache is empty but file exists with {file_size} bytes. This might indicate a loading issue.")
                    return
            with open(cache_file_path, "w") as f:
                json.dump(self._cache, f)
            logging.debug(f"[phash_cache] Saved {len(self._cache)} hashes to {cache_file_path}")
        except Exception as e:
            logging.warning(f"[phash_cache] Failed to save cache: {e}", exc_info=True)

    def get_hash(self, path: str) -> str | None:
        """Get cached hash for a file, or None if not cached or invalid."""
        self._load_cache()
        cache_key = _phash_cache_key(path)
        result = self._cache.get(cache_key)
        if result:
            logging.debug(f"[phash_cache] Cache HIT for {os.path.basename(path)} (key: {cache_key[:16]}...)")
        else:
            logging.debug(f"[phash_cache] Cache MISS for {os.path.basename(path)} (key: {cache_key[:16]}..., cache has {len(self._cache)} entries)")
        return result

    def set_hash(self, path: str, hash_string: str):
        """Cache a hash for a file."""
        self._load_cache()  # Ensure cache is loaded before adding
        cache_key = _phash_cache_key(path)
        old_size = len(self._cache)
        self._cache[cache_key] = hash_string
        # Auto-save periodically (every 100 additions) or on explicit save
        # Only save if we actually added a new entry (not updating existing)
        if len(self._cache) > old_size and len(self._cache) % 100 == 0:
            self._save_cache()

    def save(self):
        """Explicitly save cache to disk."""
        self._save_cache()

    def clear(self):
        """Clear all cached hashes."""
        self._cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
        except Exception as e:
            logging.warning(f"[phash_cache] Failed to clear cache: {e}")

