import math

from src.view_helpers import map_bbox_to_thumbnail


def test_map_bbox_to_thumbnail_basic():
    bbox = (10, 20, 110, 220)
    det_w, det_h = 200, 400
    img_w, img_h = 100, 200
    rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h)
    assert (rx1, ry1, rx2, ry2) == (
        int(10 * (img_w / det_w)),
        int(20 * (img_h / det_h)),
        int(110 * (img_w / det_w)),
        int(220 * (img_h / det_h)),
    )


def test_map_bbox_with_offset_and_non_integer_scaling():
    bbox = (5.5, 7.25, 55.75, 87.5)
    det_w, det_h = 333, 777
    img_w, img_h = 120, 120
    offset_x, offset_y = 3, 5
    rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h, offset_x, offset_y)

    # compute expected via same math, but ensure int() semantics
    sx = float(img_w) / float(det_w)
    sy = float(img_h) / float(det_h)
    exp = (
        int(bbox[0] * sx) + int(offset_x),
        int(bbox[1] * sy) + int(offset_y),
        int(bbox[2] * sx) + int(offset_x),
        int(bbox[3] * sy) + int(offset_y),
    )
    assert (rx1, ry1, rx2, ry2) == exp
