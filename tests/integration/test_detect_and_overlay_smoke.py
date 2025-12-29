from pathlib import Path

import src.object_detection as od
from src.view_helpers import update_label_icon
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QGuiApplication


def test_detect_and_overlay_smoke(monkeypatch, tmp_path):
    # Monkeypatch detect_objects to return a bbox
    sample = {'class': 'cat', 'confidence': 0.88, 'bbox': [10, 20, 90, 180], 'det_w': 200, 'det_h': 400}
    monkeypatch.setattr(od, 'detect_objects', lambda path, *a, **k: [sample])

    # Create label and call detect+overlay path
    # Ensure a QGuiApplication exists for QPixmap
    if QGuiApplication.instance() is None:
        QGuiApplication([])
    pm = QPixmap(120, 120)
    pm.fill()
    class L:
        def __init__(self):
            self._saved = None
        def setPixmap(self, pm):
            self._saved = pm
    label = L()
    label.original_pixmap = pm
    label._overlay_image_offset = (0, 0, 120, 120)

    # Simulate loading detection and drawing
    dets = od.detect_objects('dummy.jpg')
    update_label_icon(label, label='keep', filename='dummy.jpg', objects=dets)

    assert hasattr(label, '_overlay_image_offset')
