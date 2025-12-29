import os
import tempfile

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
import logging

log = logging.getLogger(__name__)

from src.viewmodel import PhotoViewModel


def process_events(ms=10):
    # Use QApplication.processEvents() instead of event loop to avoid hanging
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        app.processEvents()
    log.debug("process_events called ms=%s app=%s", ms, bool(app))
    # Small sleep to allow async operations to complete
    import time
    time.sleep(ms / 1000.0)


def test_viewmodel_state_snapshot_selection():
    QCoreApplication.instance()  # ensure app exists
    with tempfile.TemporaryDirectory() as tmpdir:
        # create a dummy image file (empty is fine for path operations)
        img_path = os.path.join(tmpdir, 'a.jpg')
        open(img_path, 'a').close()

        vm = PhotoViewModel(tmpdir, max_images=10)

        snapshots = []
        vm.browser_state_changed.connect(lambda s: snapshots.append(s))

        # Simulate images discovered without spinning up thread
        vm.images = ['a.jpg']
        vm._emit_state_snapshot()  # initial snapshot
        assert len(snapshots) == 1
        assert snapshots[-1].images == ['a.jpg']
        assert not snapshots[-1].has_selection

        vm.select_image('a.jpg')
        process_events()

        # After selection we should have another snapshot
        assert len(snapshots) >= 2
        last = snapshots[-1]
        assert last.primary.endswith('a.jpg')
        assert last.has_selection is True
        assert last.rating == 0
        assert last.selected and last.selected[0].endswith('a.jpg')
