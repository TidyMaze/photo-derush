import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap
from src import view_helpers


@pytest.fixture(scope='module', autouse=True)
def qapp():
    """Create a QApplication for QPixmap/QPainter usage in tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class DummyLabel:
    pass


def test_update_label_icon_requires_original_pixmap():
    lbl = DummyLabel()
    # no original_pixmap attribute
    with pytest.raises(ValueError):
        view_helpers.update_label_icon(lbl, 'keep', filename='x')


def test_update_label_icon_requires_overlay_offset():
    lbl = DummyLabel()
    lbl.original_pixmap = QPixmap(64, 64)
    # missing _overlay_image_offset
    with pytest.raises(ValueError):
        view_helpers.update_label_icon(lbl, 'keep', filename='x')


def test_map_bbox_to_thumbnail_scaling():
    # det image 800x600 mapped into displayed 200x150 at offset (10,20)
    bbox = (100.0, 50.0, 300.0, 250.0)
    rx1, ry1, rx2, ry2 = view_helpers.map_bbox_to_thumbnail(bbox, 800, 600, 200, 150, 10, 20)
    # expected scale factors sx=200/800=0.25, sy=150/600=0.25
    assert (rx1, ry1, rx2, ry2) == (int(100*0.25)+10, int(50*0.25)+20, int(300*0.25)+10, int(250*0.25)+20)
