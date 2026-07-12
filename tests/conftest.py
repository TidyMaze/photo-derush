import os

# Ensure Qt runs in offscreen mode for headless CI/test environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure a Qt application instance exists for timers, signals and queued
# cross-thread deliveries to work during tests. Use QApplication so
# widgets and QTimers behave correctly in offscreen mode.
try:
	from PySide6.QtWidgets import QApplication
	qapp = QApplication.instance() or QApplication([])
except Exception:  # pragma: no cover - defensive for non-Qt environments
	QApplication = None
	qapp = None

# Configure logging for tests so debug information from src modules
# (TaskRunner, AutoLabelManager, etc.) is visible when tests hang.
import logging
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
root = logging.getLogger()
if not root.handlers:
	root.addHandler(handler)
root.setLevel(logging.DEBUG)

# Reduce verbosity for noisy external libraries
logging.getLogger('PIL').setLevel(logging.WARNING)


import pytest
from PySide6.QtCore import QThreadPool


@pytest.fixture(autouse=True)
def wait_for_qt_tasks(request):
	"""Ensure any queued Qt background tasks complete after each test.

	Many tests submit QRunnable jobs to QThreadPool. If tests finish
	while background jobs are still running they can cause cross-test
	interference or leave threads alive producing RuntimeErrors. Wait
	briefly for the global pool to finish work to keep tests deterministic.
	"""
	yield
	pool = QThreadPool.globalInstance()
	try:
		logging.getLogger(__name__).debug("Waiting for QThreadPool to finish (timeout=2000ms)")
		pool.waitForDone(2000)  # wait up to 2s for pending QRunnables
		logging.getLogger(__name__).debug("QThreadPool.waitForDone finished")
	except Exception as e:
		logging.getLogger(__name__).warning("Error while waiting for QThreadPool: %s", e)


# Centralized tracking and cleanup of PhotoViewModel instances to prevent WinError 32 locked files.
_active_viewmodels = []

@pytest.fixture(autouse=True)
def cleanup_viewmodels():
    import sys
    if "src.viewmodel" in sys.modules:
        from src.viewmodel import PhotoViewModel
        if not hasattr(PhotoViewModel, "_original_init"):
            PhotoViewModel._original_init = PhotoViewModel.__init__
            def _tracked_init(self, *args, **kwargs):
                _active_viewmodels.append(self)
                PhotoViewModel._original_init(self, *args, **kwargs)
            PhotoViewModel.__init__ = _tracked_init
    yield
    while _active_viewmodels:
        vm = _active_viewmodels.pop()
        try:
            vm.cleanup()
        except Exception:
            pass


# Prevent garbage collection of any created QApplication instance across tests.
_global_qapp = None
try:
    from PySide6.QtWidgets import QApplication
    _original_qapp_init = QApplication.__init__
    def _tracked_qapp_init(self, *args, **kwargs):
        global _global_qapp
        _global_qapp = self
        _original_qapp_init(self, *args, **kwargs)
    QApplication.__init__ = _tracked_qapp_init
except Exception:
    pass


