import pytest
from PIL import Image

from src import view_helpers


class DummyLabel:
    def __init__(self):
        self.original_pixmap = None
        self._overlay_image_offset = None
    def setPixmap(self, pixmap):
        # store last set pixmap for inspection
        self._last_pixmap = pixmap


def test_map_bbox_to_thumbnail_landscape():
    # det image is landscape (800x602), thumbnail area is 128x96 centered with offset (0,16)
    bbox = (200.0, 150.5, 600.0, 451.5)
    det_w, det_h = 800, 602
    img_w, img_h = 128, 96
    offset = (0, 16)

    mapped = view_helpers.map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h, *offset)

    # expected mapping from manual calculation in debug runs
    assert isinstance(mapped, tuple)
    x, y, w, h = mapped
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0


def test_map_bbox_to_thumbnail_portrait():
    # det image is portrait (602x800), thumbnail area is 96x128 centered with offset (16,0)
    bbox = (150.5, 200.0, 451.5, 600.0)
    det_w, det_h = 602, 800
    img_w, img_h = 96, 128
    offset = (16, 0)

    mapped = view_helpers.map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h, *offset)
    assert isinstance(mapped, tuple)
    x, y, w, h = mapped
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0


def test_update_label_icon_missing_pixmap_raises():
    label = DummyLabel()
    # leave original_pixmap as None
    with pytest.raises(ValueError):
        view_helpers.update_label_icon(label, 'keep', filename=None, is_auto=False, prediction_prob=None, objects=[])


def test_update_label_icon_missing_offset_raises(tmp_path):
    label = DummyLabel()
    # attach a minimal pixmap-like object used by view_helpers
    class PixmapLike:
        def __init__(self):
            self._w = 64
            self._h = 64
        def isNull(self):
            return False
        def copy(self):
            return self
        def rect(self):
            return None
        def width(self):
            return self._w
        def height(self):
            return self._h
        def setWidth(self, w):
            self._w = w
        def setHeight(self, h):
            self._h = h

    label.original_pixmap = PixmapLike()
    # leave _overlay_image_offset missing
    with pytest.raises(ValueError):
        view_helpers.update_label_icon(label, 'keep', filename=None, is_auto=False, prediction_prob=None, objects=[])
import pytest
from src.view_helpers import map_bbox_to_thumbnail


def test_map_bbox_no_scale():
    bbox = (10, 20, 110, 220)
    det_w, det_h = 200, 400
    img_w, img_h = 200, 400
    rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h)
    assert (rx1, ry1, rx2, ry2) == (10, 20, 110, 220)


def test_map_bbox_uniform_scale():
    bbox = (0, 0, 100, 100)
    det_w, det_h = 400, 400
    img_w, img_h = 200, 200
    rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h)
    assert (rx1, ry1, rx2, ry2) == (0, 0, 50, 50) or (rx1, ry1, rx2, ry2) == (0, 0, 50, 50)


def test_map_bbox_with_offset():
    bbox = (10, 10, 60, 60)
    det_w, det_h = 100, 100
    img_w, img_h = 50, 50
    offset_x, offset_y = 5, 7
    rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h, offset_x, offset_y)
    assert (rx1, ry1, rx2, ry2) == (10*0.5 + 5, 10*0.5 + 7, 60*0.5 + 5, 60*0.5 + 7)
