import os
from pathlib import Path

from PySide6.QtGui import QPixmap, QGuiApplication

from src.view_helpers import update_label_icon


class DummyLabel:
    def __init__(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self._overlay_image_offset = (0, 0, pixmap.width(), pixmap.height())
        self._saved = None

    def setPixmap(self, pm: QPixmap):
        # Save a copy to disk for manual inspection if needed
        self._saved = pm


def test_update_label_icon_draws_boxes(tmp_path):
    # Ensure a QGuiApplication exists for QPixmap
    if QGuiApplication.instance() is None:
        QGuiApplication([])
    # Create a simple pixmap
    pm = QPixmap(120, 120)
    pm.fill()
    label = DummyLabel(pm)

    # Prepare objects: single bbox
    objs = [{'class': 'dog', 'confidence': 0.9, 'bbox': [10, 20, 50, 60], 'det_w': 100, 'det_h': 100}]

    # Should not raise
    update_label_icon(label, label='keep', filename='x.jpg', objects=objs)

    # Saved pixmap exists
    assert label._saved is not None
    # Optionally write file for debugging
    out = tmp_path / 'out.png'
    label._saved.save(str(out))
    assert out.exists()
