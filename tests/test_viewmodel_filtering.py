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


def test_viewmodel_filter_initial_and_rating_cycle():
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        app = QCoreApplication.instance() or QCoreApplication([])
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['a.jpg', 'b.jpg', 'c.jpg']:
                open(os.path.join(tmpdir, name), 'a').close()
            vm = PhotoViewModel(tmpdir, max_images=10)
            snapshots = []
            vm.browser_state_changed.connect(lambda s: snapshots.append(s))
            vm.load_images()
            process_events()
            # Last snapshot should contain all images unfiltered
            all_imgs = {'a.jpg', 'b.jpg', 'c.jpg'}
            assert set(snapshots[-1].images) == all_imgs
            assert set(snapshots[-1].filtered_images) == all_imgs
            # Set ratings
            vm.select_image('b.jpg'); process_events()
            vm.set_rating(4); process_events()
            vm.select_image('c.jpg'); process_events()
            vm.set_rating(5); process_events()
            # Apply rating filter >=5
            vm.set_filter_rating(5); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'c.jpg'}
            assert last.filtered_count == 1
            # Relax filter to >=4
            vm.set_filter_rating(4); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'b.jpg', 'c.jpg'}
            # Clear filters
            vm.clear_filters(); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == all_imgs


def test_viewmodel_filter_tag_and_case_insensitive():
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        QCoreApplication.instance()  # ensure app exists
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['a.jpg', 'b.jpg', 'c.jpg']:
                open(os.path.join(tmpdir, name), 'a').close()
            vm = PhotoViewModel(tmpdir, max_images=10)
            snapshots = []
            vm.browser_state_changed.connect(lambda s: snapshots.append(s))
            vm.load_images(); process_events()
            # Assign tags
            vm.select_image('a.jpg'); process_events()
            vm.set_tags(['dog', 'Outdoor']); process_events()
            vm.select_image('b.jpg'); process_events()
            vm.set_tags(['cat']); process_events()
            # Filter by tag 'cat'
            vm.set_filter_tag('cat'); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'b.jpg'}
            # Filter by tag case-insensitive 'DOG'
            vm.set_filter_tag('DOG'); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'a.jpg'}
            # Combine rating + tag: give a rating 4 and set rating filter 4
            vm.set_filter_rating(4); process_events()
            last = snapshots[-1]
            # a.jpg currently has rating 0, so not included when rating filter active
            assert 'a.jpg' not in last.filtered_images
            # Increase rating then ensure it appears
            vm.set_filter_tag('dog'); process_events()  # maintain tag filter
            vm.select_image('a.jpg'); process_events()
            vm.set_rating(4); process_events()
            last = snapshots[-1]
            assert 'a.jpg' in last.filtered_images
            # Clear
            vm.clear_filters(); process_events()
            last = snapshots[-1]
            assert set(last.filtered_images) == {'a.jpg', 'b.jpg', 'c.jpg'}
