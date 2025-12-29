import pytest
from src.view_helpers import select_overlay_detections


def test_select_simple_less_than_cap():
    objs = [('cat', 0.7), ('dog', 0.6)]
    mode, items = select_overlay_detections(objs, high_conf_threshold=0.8, high_conf_cap=5)
    assert mode == 'simple'
    assert len(items) == 2


def test_select_simple_more_high_than_cap():
    objs = [('a', 0.95), ('b', 0.93), ('c', 0.92), ('d', 0.91), ('e', 0.9), ('f', 0.89)]
    mode, items = select_overlay_detections(objs, high_conf_threshold=0.8, high_conf_cap=5)
    assert mode == 'simple'
    assert len(items) == 5
    # top item should be 'a' with highest score
    assert items[0]['name'] == 'a'


def test_select_bbox_less_than_cap():
    dets = [ {'bbox': [0,0,10,10], 'confidence': 0.6, 'class':'cat'}, {'bbox': [1,1,12,12], 'confidence': 0.7, 'class':'dog'} ]
    mode, items = select_overlay_detections(dets, high_conf_threshold=0.8, high_conf_cap=5)
    assert mode == 'bbox'
    assert len(items) == 2


def test_select_bbox_more_high_than_cap():
    dets = [ {'bbox':[0,0,1,1], 'confidence': s, 'class':'obj'} for s in [0.95,0.94,0.93,0.92,0.91,0.9] ]
    mode, items = select_overlay_detections(dets, high_conf_threshold=0.8, high_conf_cap=3)
    assert mode == 'bbox'
    assert len(items) == 3
    # Ensure descending order
    confs = [float(x['confidence']) for x in items]
    assert confs == sorted(confs, reverse=True)
