import os
import pytest
from src import object_detection as od

TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_images')


def _sample_images(limit=2):
    if not os.path.isdir(TEST_DIR):
        pytest.skip('No test_images directory available')
    imgs = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        pytest.skip('No test images found')
    return imgs[:limit]


@pytest.mark.parametrize('backend', ['fasterrcnn', 'ssdlite'])
def test_detect_objects_basic(backend):
    # test detection works and returns expected keys; skip if backend not available
    os.environ['DETECTION_BACKEND'] = backend
    # reload module to pick up env change
    import importlib
    importlib.reload(od)

    images = _sample_images()
    for img in images:
        dets = od.detect_objects(img, confidence_threshold=0.01)
        assert isinstance(dets, list)
        # allow zero detections but ensure structure for any returned detections
        for d in dets:
            assert isinstance(d, dict)
            assert 'class' in d and 'confidence' in d and 'bbox' in d
            assert 'det_w' in d and 'det_h' in d
            assert len(d['bbox']) == 4


def test_get_available_classes():
    classes = od.get_available_classes()
    assert isinstance(classes, list)
    assert 'person' in classes