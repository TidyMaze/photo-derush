from __future__ import annotations

"""Generic Qt-based background task runner with progress reporting.

Provides a lightweight abstraction to move heavy work off the GUI thread using
QThreadPool + QRunnable while emitting structured progress signals.

Design goals:
- Simple API for fire-and-forget tasks.
- Task progress emits (name, current, total) where total==0 => indeterminate.
- Safe exception logging; failures propagate via finished signal with success=False.
- Allows cooperative progress updates from worker via a ProgressReporter passed
  into the callable.
"""
import logging
import time
import traceback
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class ProgressReporter:
    __slots__ = ("_task", "_current", "_total", "_name", "_detail")

    def __init__(self, task: _RunnableTask, name: str):
        self._task = task
        self._name = name
        self._current = 0
        self._total = 0
        self._detail = ""

    def set_total(self, total: int):
        total = int(total) if total and total > 0 else 0
        self._total = total
        self._task._emit_progress(self._name, self._current, self._total, self._detail)

    def advance(self, delta: int = 1):
        self._current += int(delta)
        if self._total and self._current > self._total:
            self._current = self._total
        self._task._emit_progress(self._name, self._current, self._total, self._detail)

    def update(self, current: int, total: int | None = None):
        self._current = max(0, int(current))
        if total is not None:
            self._total = int(total) if total > 0 else 0
        self._task._emit_progress(self._name, self._current, self._total, self._detail)

    def detail(self, text: str):
        self._detail = text or ""
        self._task._emit_progress(self._name, self._current, self._total, self._detail)


class _RunnableTask:
    def __init__(self, name: str, fn: Callable[[ProgressReporter], None], runner: TaskRunner):
        self._name = name
        self._fn = fn
        self._runner = runner
        # OPTIMIZATION: Debounce progress updates to reduce signal emissions
        # Use time-based rate limiting instead of Qt timer (works from any thread)
        self._pending_progress = None
        self._last_emit_time = 0.0
        self._emit_debounce_sec = 0.05  # 50ms debounce

    def _emit_progress(self, name: str, current: int, total: int, detail: str):
        # OPTIMIZATION: Batch progress updates with time-based debounce (thread-safe)
        self._pending_progress = (name, current, total, detail)
        
        # Check if enough time has passed since last emit
        now = time.perf_counter()
        if now - self._last_emit_time >= self._emit_debounce_sec:
            self._flush_progress()
            self._last_emit_time = now
    
    def _flush_progress(self):
        """Emit the pending progress update."""
        if self._pending_progress is None:
            return
        name, current, total, detail = self._pending_progress
        self._pending_progress = None
        try:
            self._runner.task_progress.emit(name, current, total, detail)
        except RuntimeError:
            # Runner QObject was deleted (app closing); swallow to avoid crashing
            logging.debug(f"TaskRunner: task_progress signal ignored, runner deleted ({name})")

    def run(self):  # noqa: D401
        logging.info(f"[TaskRunner] Task '{self._name}' started")
        try:
            self._runner.task_started.emit(self._name)
        except RuntimeError:
            logging.debug(f"TaskRunner: task_started signal ignored, runner deleted ({self._name})")
        ok = True
        t0 = time.perf_counter()
        try:
            reporter = ProgressReporter(self, self._name)
            self._fn(reporter)
            # OPTIMIZATION: Flush any pending progress before task completes
            self._flush_progress()
            elapsed = time.perf_counter() - t0
            logging.info(f"[TaskRunner] Task '{self._name}' completed successfully in {elapsed:.3f}s")
        except Exception as e:  # pragma: no cover - defensive path
            ok = False
            elapsed = time.perf_counter() - t0
            logging.error(f"Task '{self._name}' failed after {elapsed:.3f}s: {e}\n{traceback.format_exc()}")
        finally:
            try:
                self._runner.task_finished.emit(self._name, ok)
            except RuntimeError:
                logging.debug(f"TaskRunner: task_finished signal ignored, runner deleted ({self._name})")


class TaskRunner(QObject):
    task_started = Signal(str)  # name
    task_progress = Signal(str, int, int, str)  # name, current, total, detail
    task_finished = Signal(str, bool)  # name, success

    def __init__(self, max_threads: int = 8):
        super().__init__()
        import queue
        import threading
        
        self._queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="TaskRunnerWorker")
        self._worker_thread.start()
        logging.info("[TaskRunner] Sequential worker thread initialized and started")

    def run(self, name: str, fn: Callable[[ProgressReporter], None]):
        if not name or not callable(fn):
            logging.error("TaskRunner.run called with invalid arguments")
            return
        logging.info(
            f"[TaskRunner] Queueing task '{name}' (current queue size: {self._queue.qsize()})"
        )
        job = _RunnableTask(name, fn, self)
        self._queue.put(job)

    def _worker_loop(self):
        import queue
        while not self._shutdown_event.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                job.run()
            except Exception as e:
                logging.exception(f"[TaskRunner] Unexpected exception in worker executing job '{getattr(job, '_name', 'unknown')}': {e}")
            finally:
                self._queue.task_done()

    # Convenience wrappers for common patterns
    def run_list(self, name: str, items, work: Callable[[object, ProgressReporter], None]):
        def _wrap(reporter: ProgressReporter):
            seq = list(items) if items else []
            reporter.set_total(len(seq))
            for it in seq:
                try:
                    work(it, reporter)
                except Exception as e:  # continue after item failure
                    logging.debug(f"Task '{name}' item error: {e}")
                reporter.advance(1)

        self.run(name, _wrap)
    
    def shutdown(self, timeout_ms: int = 3000):
        """Shutdown TaskRunner, waiting for active tasks to complete."""
        import queue
        logging.info("[TaskRunner] Shutting down worker thread")
        self._shutdown_event.set()
        # Wait for thread to finish
        self._worker_thread.join(timeout=timeout_ms / 1000.0)
        # Clear any remaining tasks in queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        logging.info("[TaskRunner] Shutdown complete")


__all__ = ["TaskRunner", "ProgressReporter"]
