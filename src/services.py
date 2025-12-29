import logging
import os
from collections.abc import Callable
from typing import Generic, TypeVar

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

T = TypeVar("T")


class _ThumbnailJob(QRunnable):
    def __init__(self, path: str, size: int, model, service):
        super().__init__()
        self.path = path
        self.size = size
        self.model = model
        self.service = service

    def run(self):
        import time
        t0 = time.perf_counter()
        if not os.path.exists(self.path):
            logging.debug(f"Thumbnail job skipped, file missing: {self.path}")
            self.service._emit_result(self.path, None)
            return
        img = self.model.load_thumbnail(self.path, self.size)
        t1 = time.perf_counter()
        self.service._emit_result(self.path, img)


class _ExifJob(QRunnable):
    def __init__(self, path: str, model, service):
        super().__init__()
        self.path = path
        self.model = model
        self.service = service

    def run(self):
        import time
        t0 = time.perf_counter()
        if not os.path.exists(self.path):
            self.service._emit_result(self.path, {})
            return
        data = self.model.load_exif(self.path)
        t1 = time.perf_counter()
        self.service._emit_result(self.path, data)


class AsyncService(QObject, Generic[T]):
    """Generic async service for loading data from model."""
    result_ready = Signal(str, object)  # path, result

    def __init__(self, model, job_class, max_concurrent: int = 8):
        super().__init__()
        self._model = model
        self._job_class = job_class
        self._callbacks: dict[str, list[Callable]] = {}
        self._inflight: set[str] = set()
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(self._pool.maxThreadCount(), max_concurrent))
        self.result_ready.connect(self._dispatch_callbacks)
        self._cancelled = False

    def cancel_all(self):
        self._callbacks.clear()
        self._inflight.clear()
        self._cancelled = True

    def request(self, path: str, callback: Callable, *args):
        if not path:
            return
        if self._cancelled:
            self._cancelled = False
        lst = self._callbacks.setdefault(path, [])
        lst.append(callback)
        if path in self._inflight:
            return
        self._inflight.add(path)
        job = self._job_class(path, *args, self._model, self)
        self._pool.start(job)

    def _emit_result(self, path: str, result: T):
        try:
            self.result_ready.emit(path, result)
        except RuntimeError:
            # QObject deleted while background job completed; ignore.
            logging.debug(f"[service] RuntimeError emitting {path}")
            return

    def _dispatch_callbacks(self, path: str, result: T):
        if path in self._inflight:
            self._inflight.remove(path)
        callbacks = self._callbacks.pop(path, [])
        for cb in callbacks:
            try:
                cb(path, result)
            except Exception as e:
                logging.error(f"Service callback error for {path}: {e}")


class ThumbnailService(AsyncService):
    thumbnail_ready = Signal(str, object)  # path, PIL Image or None

    def __init__(self, model, max_concurrent: int = 8):
        super().__init__(model, _ThumbnailJob, max_concurrent)
        self.thumbnail_ready = self.result_ready

    def request_thumbnail(self, path: str, size: int, callback: Callable[[str, object], None]):
        import time
        t0 = time.perf_counter()
        self.request(path, callback, size)
        t1 = time.perf_counter()


class ExifService(AsyncService):
    exif_ready = Signal(str, dict)  # path, exif dict

    def __init__(self, model, max_concurrent: int = 4):
        super().__init__(model, _ExifJob, max_concurrent)
        self.exif_ready = self.result_ready

    def request_exif(self, path: str, callback: Callable[[str, dict], None]):
        self.request(path, callback)
