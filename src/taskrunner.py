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


class _RunnableTask(QRunnable):
    def __init__(self, name: str, fn: Callable[[ProgressReporter], None], runner: TaskRunner):
        super().__init__()
        self._name = name
        self._fn = fn
        self._runner = runner

    def _emit_progress(self, name: str, current: int, total: int, detail: str):
        try:
            self._runner.task_progress.emit(name, current, total, detail)
        except RuntimeError:
            # Runner QObject was deleted (app closing); swallow to avoid crashing
            logging.debug(f"TaskRunner: task_progress signal ignored, runner deleted ({name})")

    def run(self):  # noqa: D401
        logging.debug(f"[TaskRunner] Task '{self._name}' started")
        try:
            self._runner.task_started.emit(self._name)
        except RuntimeError:
            logging.debug(f"TaskRunner: task_started signal ignored, runner deleted ({self._name})")
        ok = True
        try:
            reporter = ProgressReporter(self, self._name)
            self._fn(reporter)
            logging.debug(f"[TaskRunner] Task '{self._name}' completed successfully")
        except Exception as e:  # pragma: no cover - defensive path
            ok = False
            logging.error(f"Task '{self._name}' failed: {e}\n{traceback.format_exc()}")
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
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(self._pool.maxThreadCount(), max_threads))

    def run(self, name: str, fn: Callable[[ProgressReporter], None]):
        if not name or not callable(fn):
            logging.error("TaskRunner.run called with invalid arguments")
            return
        logging.debug(
            f"[TaskRunner] Submitting task '{name}' (pool: {self._pool.activeThreadCount()}/{self._pool.maxThreadCount()} active)"
        )
        job = _RunnableTask(name, fn, self)
        self._pool.start(job)

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
        try:
            if self._pool:
                # Wait for active tasks with timeout
                self._pool.waitForDone(timeout_ms)
                # Clear any remaining tasks
                self._pool.clear()
                logging.info(f"[TaskRunner] Shutdown complete (pool: {self._pool.activeThreadCount()} active threads)")
        except Exception as e:
            logging.warning(f"[TaskRunner] Error during shutdown: {e}")


__all__ = ["TaskRunner", "ProgressReporter"]
