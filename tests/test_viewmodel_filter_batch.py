import os
import tempfile
from unittest.mock import patch

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


def test_viewmodel_batch_set_filters_and_idempotent_snapshot():
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        QCoreApplication.instance()  # ensure app exists
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['a.jpg', 'b.jpg', 'c.jpg']:
                open(os.path.join(tmpdir, name), 'a').close()
            vm = PhotoViewModel(tmpdir, max_images=10)
            snapshots = []
            vm.browser_state_changed.connect(lambda s: snapshots.append(s))
            vm.load_images(); process_events()
            # Assign ratings / tags
            vm.select_image('a.jpg'); process_events(); vm.set_tags(['cat']); vm.set_rating(3); process_events()
            vm.select_image('b.jpg'); process_events(); vm.set_tags(['cat']); vm.set_rating(4); process_events()
            vm.select_image('c.jpg'); process_events(); vm.set_tags(['dog']); vm.set_rating(5); process_events()
            base_len = len(snapshots)
            # Batch set rating>=4 and tag=cat -> expect only b.jpg
            vm.set_filters(rating=4, tag='cat'); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'b.jpg'}
            assert vm.active_filters() == {'rating': 4, 'tag': 'cat', 'date': ''}
            new_len = len(snapshots)
            assert new_len == base_len + 1  # single emission
            # Idempotent call (no change) should not emit
            vm.set_filters(rating=4, tag='cat'); process_events()
            assert len(snapshots) == new_len
            # Change only tag maintaining rating threshold to include none (tag=dog with rating>=4 -> only c.jpg qualifies)
            vm.set_filters(tag='dog'); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'c.jpg'}
            # Clear with batch call
            vm.set_filters(rating=0, tag=''); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'a.jpg', 'b.jpg', 'c.jpg'}
