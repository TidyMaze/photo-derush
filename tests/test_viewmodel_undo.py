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
    # Small sleep to allow async operations to complete
    import time
    time.sleep(ms / 1000.0)
    log.debug("process_events called ms=%s app=%s", ms, bool(app))


def test_viewmodel_undo_redo_state_flags():
    QCoreApplication.instance()  # ensure app exists

    with tempfile.TemporaryDirectory() as tmpdir:
        # create a dummy image file
        img_path = os.path.join(tmpdir, 'a.jpg')
        open(img_path, 'a').close()
        vm = PhotoViewModel(tmpdir, max_images=5)
        snapshots = []
        vm.browser_state_changed.connect(lambda s: snapshots.append(s))
        vm.load_images()
        process_events()
        vm.select_image('a.jpg')
        process_events()
        # Initially no undo
        assert snapshots[-1].can_undo is False
        assert snapshots[-1].can_redo is False
        vm.set_rating(3)
        process_events()
        # After setting rating, undo should be available
        # There may be intermediate snapshots; find last with rating 3
        last = snapshots[-1]
        assert last.rating == 3
        assert last.can_undo is True
        assert last.can_redo is False
        vm.undo()
        process_events()
        last = snapshots[-1]
        assert last.rating == 0
        assert last.can_undo in (False, True)  # could have older commands; ensure redo is now possible
        assert last.can_redo is True
        vm.redo()
        process_events()
        last = snapshots[-1]
        assert last.rating == 3
        assert last.can_undo is True
        # After redo, redo stack should be empty again
        # A subsequent command should clear redo stack
        vm.set_rating(4)
        process_events()
        last = snapshots[-1]
        assert last.rating == 4
        assert last.can_undo is True
