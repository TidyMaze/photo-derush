import os

# Ensure Qt runs in offscreen mode for headless CI/test environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure a Qt application instance exists for timers, signals and queued
# cross-thread deliveries to work during tests. Use QApplication so
# widgets and QTimers behave correctly in offscreen mode.
try:
	from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - defensive for non-Qt environments
	QApplication = None

# Do NOT create a global QApplication here. Creating it changes internal
# code paths (e.g. making the app non-headless) and causes tests that rely
# on synchronous/headless behavior to schedule GUI timers instead, which
# can lead to hangs when tests don't pump the event loop. Tests that need
# a QApplication should create it explicitly.

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
