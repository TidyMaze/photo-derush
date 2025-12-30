"""Lazy image metadata loading with threading & LRU caching.

Provides non-blocking EXIF & thumbnail extraction with cancel support.
Uses thread pool for parallelization and LRU cache to minimize re-extraction.
Qt signals marshal callbacks back to main thread for thread-safe UI updates.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from PySide6.QtCore import QObject, Signal, Qt

from .cache_config import is_cache_disabled


class _LazyLoaderSignals(QObject):
    """Qt signals for marshaling thread pool callbacks to main thread."""

    exif_loaded = Signal(str, dict)  # (path, exif_dict)
    thumbnail_loaded = Signal(str, object)  # (path, thumbnail_image)
    detection_done = Signal(str, list)  # (path, detections_list)


class LazyImageLoader:
    """Lazy loader for EXIF data and thumbnails with thread pool & LRU cache."""

    def __init__(self, model, max_workers: int = 4, cache_size: int = 256):
        """Initialize loader.

        Args:
            model: ImageModel instance (has load_exif, load_thumbnail methods)
            max_workers: Thread pool size
            cache_size: LRU cache size per resource type
        """
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self._pending_tasks: dict[str, list] = {}
        self._cancelled = False

        # Create Qt signal bridge for thread-safe callbacks
        self._signals = _LazyLoaderSignals()

        # Create LRU-wrapped methods
        self._cached_exif = lru_cache(maxsize=cache_size)(self._load_exif_uncached)
        self._cached_thumb = lru_cache(maxsize=cache_size)(self._load_thumb_uncached)

    def _load_exif_uncached(self, path: str) -> dict[Any, Any]:
        """Load EXIF without cache (for LRU wrapping)."""
        result = self.model.load_exif(path)
        return dict(result) if isinstance(result, dict) else {}

    def _load_thumb_uncached(self, path: str) -> object | None:
        """Load thumbnail without cache (for LRU wrapping)."""
        result = self.model.load_thumbnail(path)
        return result  # type: ignore[return-value, no-any-return]

    def get_exif_lazy(self, path: str, callback: Callable[[str, dict], None]) -> None:
        """Load EXIF asynchronously (callback runs in main thread via Qt signal).

        Args:
            path: Image path
            callback: Called with (path, exif_dict) when done
        """
        if self._cancelled:
            logging.debug("[loader] Ignoring request (cancelled)")
            return

        # Connect signal to callback once
        self._signals.exif_loaded.connect(
            lambda p, exif: callback(p, exif) if p == path else None,
            type=Qt.ConnectionType.QueuedConnection,  # main thread
        )

        def _load_and_emit():
            try:
                if is_cache_disabled():
                    result = self._load_exif_uncached(path)
                else:
                    result = self._cached_exif(path)
                if not self._cancelled:
                    self._signals.exif_loaded.emit(path, result)
            except Exception as e:
                logging.warning("[loader] EXIF load failed for %s: %s", path, e)
                if not self._cancelled:
                    self._signals.exif_loaded.emit(path, {})

        self.executor.submit(_load_and_emit)

    def get_thumbnail_lazy(self, path: str, callback: Callable[[str, object | None], None]) -> None:
        """Load thumbnail asynchronously (callback runs in main thread via Qt signal).

        Args:
            path: Image path
            callback: Called with (path, thumbnail_image) when done
        """
        if self._cancelled:
            logging.debug("[loader] Ignoring request (cancelled)")
            return

        # Connect signal to callback once
        self._signals.thumbnail_loaded.connect(
            lambda p, thumb: callback(p, thumb) if p == path else None,
            type=Qt.ConnectionType.QueuedConnection,  # main thread
        )

        def _load_and_emit():
            try:
                if is_cache_disabled():
                    result = self._load_thumb_uncached(path)
                else:
                    result = self._cached_thumb(path)
                if not self._cancelled:
                    self._signals.thumbnail_loaded.emit(path, result)
            except Exception as e:
                logging.warning("[loader] Thumbnail load failed for %s: %s", path, e)
                if not self._cancelled:
                    self._signals.thumbnail_loaded.emit(path, None)

        self.executor.submit(_load_and_emit)

    def batch_load_exif(
        self,
        paths: list[str],
        on_progress: Callable[[int, int], None],
        on_complete: Callable[[], None],
    ) -> None:
        """Batch load EXIF for multiple paths with progress callback.

        Args:
            paths: List of image paths
            on_progress: Called with (completed, total) on each completion
            on_complete: Called when all done
        """
        if self._cancelled:
            return

        total = len(paths)

        def _batch_worker():
            completed = 0
            try:
                for path in paths:
                    if self._cancelled:
                        break
                    if is_cache_disabled():
                        self._load_exif_uncached(path)
                    else:
                        self._cached_exif(path)
                    completed += 1
                    on_progress(completed, total)
            finally:
                on_complete()

        # Run in background thread
        thread = threading.Thread(target=_batch_worker, daemon=True)
        thread.start()

    def clear_cache(self) -> None:
        """Clear LRU caches."""
        with self._lock:
            self._cached_exif.cache_clear()
            self._cached_thumb.cache_clear()
        logging.info("[loader] Caches cleared")

    def cancel(self) -> None:
        """Cancel all pending operations."""
        with self._lock:
            self._cancelled = True
        logging.info("[loader] Cancelled pending operations")

    def resume(self) -> None:
        """Resume operations after cancel."""
        with self._lock:
            self._cancelled = False

    def shutdown(self, wait: bool = True) -> None:
        """Shut down loader safely.

        This method attempts to stop further callbacks from worker threads
        being delivered to the Qt main thread during application shutdown.
        It marks the loader cancelled, replaces the Qt signal bridge (breaking
        existing connections), clears caches, and then shuts down the executor.

        The executor is shutdown with `wait=False` to avoid blocking the UI
        thread; callers that need a blocking shutdown can call this method and
        then join any remaining application-level threads as appropriate.
        """
        # Prevent new work from emitting callbacks
        with self._lock:
            self._cancelled = True

            # Replace the signal bridge so existing connected callbacks are
            # detached. Keep a reference to the old signals so we can try to
            # disconnect them (best-effort).
            old_signals = getattr(self, "_signals", None)
            try:
                self._signals = _LazyLoaderSignals()
            except Exception:
                # In the unlikely event creation fails, fall back to keeping
                # the existing signals but still mark cancelled.
                logging.exception("[loader] Failed to replace signals during shutdown")

        # Best-effort: disconnect old signals to remove queued callbacks.
        if old_signals is not None:
            try:
                old_signals.exif_loaded.disconnect()
            except Exception:
                logging.exception("Error disconnecting old exif_loaded signal during shutdown; continuing")
            try:
                old_signals.thumbnail_loaded.disconnect()
            except Exception:
                logging.exception("Error disconnecting old thumbnail_loaded signal during shutdown; continuing")

        # Clear caches to release references
        try:
            self.clear_cache()
        except Exception:
            logging.exception("[loader] Exception while clearing caches")

        # Shutdown executor without waiting to avoid blocking the GUI thread.
        # We honor the `wait` argument by logging but still perform a
        # non-blocking shutdown here because waiting can deadlock if worker
        # threads are waiting to post to the Qt main loop which is shutting
        # down.
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            logging.exception("[loader] Exception during executor.shutdown")

        logging.info("[loader] Shutdown requested (cancelled=%s, wait=%s)", self._cancelled, wait)

    def cache_info(self) -> dict:
        """Return cache statistics."""
        return {
            "exif_hits": self._cached_exif.cache_info().hits,
            "exif_misses": self._cached_exif.cache_info().misses,
            "thumb_hits": self._cached_thumb.cache_info().hits,
            "thumb_misses": self._cached_thumb.cache_info().misses,
        }


__all__ = ["LazyImageLoader"]
