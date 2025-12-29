import os
import pytest

from src import object_detection as od


def _sample_images(limit=2):
    base = os.path.join(os.path.dirname(__file__), '..', 'test_images')
    if not os.path.isdir(base):
        pytest.skip('No test_images/ directory present')
    imgs = []
    for entry in os.listdir(base):
        if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
            imgs.append(os.path.join(base, entry))
            if len(imgs) >= limit:
                break
    if not imgs:
        pytest.skip('No images found in test_images/')
    return imgs


@pytest.mark.parametrize('backend', ['fasterrcnn', 'ssdlite', 'yolov8'])
def test_detect_backends(tmp_path, backend):
    # Skip yolov8 if ultralytics not installed
    if backend == 'yolov8':
        try:
            import ultralytics  # noqa: F401
        except Exception:
            pytest.skip('ultralytics not installed; skipping yolov8 test')

    # Set backend via env and reload module-level constant
    os.environ['DETECTION_BACKEND'] = backend
    # Reload module to pick up env var change
    import importlib
    importlib.reload(od)

    imgs = _sample_images(limit=2)
    for img in imgs:
        dets = od.detect_objects(img, confidence_threshold=0.1, device='cpu', max_size=800)
        # Detections should be a list; entries should contain required keys
        assert isinstance(dets, list)
        for d in dets:
            assert 'class' in d
            assert 'confidence' in d
            assert 'bbox' in d
            assert 'det_w' in d and 'det_h' in d
