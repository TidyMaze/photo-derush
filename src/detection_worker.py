from __future__ import annotations

"""Simplified detection worker process.

Loads YOLOv8 in a separate process to avoid crashes in the main app.
"""
import logging
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Dict, Optional

DETECTION_BACKEND = os.environ.get("DETECTION_BACKEND", "yolov8")


def _worker_main(in_q: mp.Queue, out_q: mp.Queue, stop_event: "mp.Event"):  # type: ignore[valid-type]
    import signal
    import traceback

    # Simple logging
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/detection_worker_{os.getpid()}.log"
        logging.basicConfig(
            filename=log_path, level=logging.INFO, format="%(asctime)s [worker %(process)d] %(levelname)s: %(message)s"
        )
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker %(process)d] %(levelname)s: %(message)s")

    # Signal handlers
    def _sig_handler(signum, frame):
        logging.error("Worker received signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    # Prevent recursive worker
    os.environ["DETECTION_WORKER"] = "0"
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from .object_detection import _load_model
        from .object_detection import detect_objects as _detect_objects

        logging.info("Worker imported detection internals")
    except Exception:
        tb = traceback.format_exc()
        logging.exception("Failed to import detection internals in worker")
        out_q.put((None, {"error": "failed to import detection internals in worker", "trace": tb}))
        return

    try:
        model, weights = _load_model("auto")
        logging.info("Worker loaded model")
    except Exception:
        tb = traceback.format_exc()
        logging.exception("Failed to load model in worker")
        out_q.put((None, {"error": "failed to load model in worker", "trace": tb}))
        return

    logging.info(f"Detection worker started with backend={DETECTION_BACKEND} pid={os.getpid()}")

    while not stop_event.is_set():  # type: ignore[attr-defined]
        try:
            req = in_q.get(timeout=0.2)
            if req is None:
                break
            request_id, image_path, kwargs = req
            logging.info(f"start request id={request_id} image={image_path}")
            start = time.time()
            try:
                raw = _detect_objects(image_path, **(kwargs or {}))
                duration = time.time() - start
                logging.info(
                    f'finished request id={request_id} image={image_path} duration={duration:.3f}s detections={len(raw) if hasattr(raw, "__len__") else "?"}'
                )
                out_q.put((request_id, {"detections": raw}))
            except Exception:
                tb = traceback.format_exc()
                logging.exception(f"detection failed for request id={request_id} image={image_path}")
                out_q.put((request_id, {"error": "detection failed", "trace": tb}))
        except KeyboardInterrupt:
            break
        except Exception:
            logging.exception("Unexpected error in detection worker loop")
            time.sleep(0.1)


class DetectionWorker:
    """Manage a background detection process.

    Usage:
        w = DetectionWorker()
        w.start()
        req_id = w.enqueue(image_path, kwargs)
        result = w.get_result(req_id, timeout=10.0)
        w.stop()
    """

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.in_q: Optional[mp.Queue] = None
        self.out_q: Optional[mp.Queue] = None
        self.stop_event: Optional[mp.Event] = None
        self.proc: Optional[mp.Process] = None
        self._next_id = 1
        # Fallback queue length tracker for platforms where `qsize()` may be
        # unavailable or unreliable (e.g., some macOS Python builds). We
        # update this counter on enqueue and when results are returned.
        self._queued_count = 0
        self._queued_lock = threading.Lock()
        # Health / restart tracking to avoid repeated failing restarts
        self._healthy = False
        self._restart_times: list[float] = []
        self._max_restarts = 3
        self._restart_window = 60.0

    def start(self):
        # If an existing process is alive, nothing to do
        if self.proc is not None and self.proc.is_alive():
            self._healthy = True
            return

        # Prevent rapid restart storms: only allow _max_restarts within _restart_window
        now = time.time()
        # prune old entries
        self._restart_times = [t for t in self._restart_times if now - t < self._restart_window]
        if len(self._restart_times) >= self._max_restarts:
            logging.warning("Detection worker exceeded restart limit; not starting new worker")
            self._healthy = False
            return

        # create queues and event with spawn context
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        try:
            self.proc = self.ctx.Process(
                target=_worker_main, args=(self.in_q, self.out_q, self.stop_event), daemon=False
            )
            self.proc.start()
        except Exception:
            logging.exception("Failed to start detection worker process")
            # mark unhealthy and record restart attempt
            self._healthy = False
            self._restart_times.append(now)
            self.proc = None
            return

        # give the process a short moment to appear alive
        time.sleep(0.2)
        if not getattr(self.proc, "is_alive", lambda: False)():
            logging.warning("Detection worker process died immediately after start")
            self._healthy = False
            self._restart_times.append(now)
            try:
                self.proc.join(timeout=0.5)
            except Exception:
                pass
            self.proc = None
            return

        # started successfully for now; mark healthy
        self._healthy = True
        self._restart_times.append(now)

    def stop(self):
        try:
            if self.proc is None:
                return
            # Signal the worker to stop
            if self.stop_event is not None:
                try:
                    self.stop_event.set()
                except Exception:
                    pass
            # Send sentinel to unblock worker if it's waiting on the input queue
            try:
                if self.in_q is not None:
                    try:
                        self.in_q.put(None)
                    except Exception:
                        pass
            except Exception:
                pass
            # Give the process a short grace period to exit
            try:
                self.proc.join(timeout=5.0)
            except Exception:
                pass
            # If still alive, attempt termination
            try:
                if getattr(self.proc, "is_alive", lambda: False)():
                    try:
                        self.proc.terminate()
                    except Exception:
                        pass
            except Exception:
                pass
        finally:
            # Clear all references so subsequent start() creates fresh queues/process
            try:
                self.in_q = None
            except Exception:
                pass
            try:
                self.out_q = None
            except Exception:
                pass
            try:
                self.stop_event = None
            except Exception:
                pass
            try:
                self.proc = None
            except Exception:
                pass
            try:
                with self._queued_lock:
                    self._queued_count = 0
            except Exception:
                pass
            try:
                self._healthy = False
            except Exception:
                pass

    def enqueue(self, image_path: str, kwargs: Optional[Dict] = None) -> int:
        req_id = self._next_id
        self._next_id += 1
        # Ensure worker is started
        if self.in_q is None or not getattr(self.proc, "is_alive", lambda: False)() or not self._healthy:
            raise RuntimeError("Detection worker not started or unhealthy")
        # Put the request on the queue and update the fallback counter.
        try:
            self.in_q.put((req_id, image_path, kwargs or {}))
        except Exception as exc:
            raise RuntimeError(f"Failed to enqueue detection request: {exc}") from exc
        try:
            with self._queued_lock:
                self._queued_count += 1
        except Exception:
            pass
        return int(req_id)  # type: ignore[return-value]

    def get_result(self, req_id: int, timeout: float = 10.0) -> Dict[str, Any]:
        # wait for matching response
        deadline = time.time() + timeout
        if self.out_q is None:
            raise RuntimeError("Detection worker not started")
        while time.time() < deadline:
            try:
                item = self.out_q.get(timeout=0.2)
            except Exception:
                continue
            if not item:
                continue
            resp_id, payload = item
            if resp_id is None:
                # worker-level error
                # mark unhealthy on worker-level failure so caller can fallback
                try:
                    self._healthy = False
                except Exception:
                    pass
                raise RuntimeError(f"Worker failed to start: {payload}")
            if resp_id == req_id:
                # A matching response means the corresponding request was
                # completed; decrement our fallback queued counter.
                try:
                    with self._queued_lock:
                        if self._queued_count > 0:
                            self._queued_count -= 1
                except Exception:
                    pass
                return payload  # type: ignore[return-value, no-any-return]
            # not my response: push back to queue (simple re-queue)
            self.out_q.put(item)
        raise TimeoutError("Timed out waiting for detection worker response")

    def queue_length(self) -> int:
        """Return approximate number of requests queued for the worker.

        This first tries to use the underlying multiprocessing queue's
        `qsize()` (may raise or be unsupported on some platforms). If that
        is unavailable, it falls back to an internal counter incremented on
        `enqueue()` and decremented when a response is received.
        """
        try:
            if self.in_q is None:
                return 0
            try:
                return self.in_q.qsize()
            except Exception:
                # Fall back
                pass
        except Exception:
            pass
        try:
            with self._queued_lock:
                return int(self._queued_count)
        except Exception:
            return 0


_global_worker: Optional[DetectionWorker] = None


def get_global_worker() -> DetectionWorker:
    global _global_worker
    if _global_worker is None:
        _global_worker = DetectionWorker()
        try:
            _global_worker.start()
        except Exception:
            # If starting the worker fails, ensure we don't keep a partially
            # initialized global reference and propagate the error so callers
            # can fall back to in-process detection.
            try:
                _global_worker = None
            except Exception:
                pass
            raise
    # If the worker failed health checks during start, do not return it.
    try:
        if not getattr(_global_worker, "_healthy", False):
            # Clean up partially-initialized worker and signal failure to caller
            try:
                _global_worker.stop()
            except Exception:
                pass
            _global_worker = None
            raise RuntimeError("Detection worker is unhealthy")
    except Exception:
        # If attribute missing or other issues, ensure we don't return a broken worker
        try:
            _global_worker = None
        except Exception:
            pass
        raise
    return _global_worker


def stop_global_worker():
    global _global_worker
    if _global_worker is not None:
        try:
            _global_worker.stop()
        except Exception:
            logging.exception("Failed to stop detection worker")
        _global_worker = None
