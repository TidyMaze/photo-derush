import os
import numpy as np
from PySide6.QtWidgets import QApplication
from photo_derush.main_window import LightroomMainWindow
from photo_derush.viewmodel import _NEW_FEATURE_NAMES

def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def get_real_image_paths(min_count=2):
    image_dir = '/Users/yannrolland/Pictures/photo-dataset'
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if len(image_paths) < min_count:
        import pytest
        pytest.skip(f"At least {min_count} .jpg images are required in {image_dir}.")
    return image_dir, image_paths

def test_metrics_status_bar():
    app = ensure_app()
    image_dir, image_paths = get_real_image_paths(2)
    img1, img2 = image_paths[:2]
    from photo_derush.main_window import compute_feature_vector
    fv1, keys1 = compute_feature_vector(os.path.join(image_dir, img1))
    fv2, keys2 = compute_feature_vector(os.path.join(image_dir, img2))
    assert fv1 is not None and fv2 is not None, "Feature extraction failed on real images."
    win = LightroomMainWindow([img1, img2], image_dir, lambda: [img1, img2])
    win.current_img_idx = 0
    win._label_current_image(0)
    win.current_img_idx = 1
    win._label_current_image(1)
    win._refresh_all_keep_probs()
    for _ in range(10):
        app.processEvents()
    msg = win.status.currentMessage()
    assert msg is None or 'acc=' in msg, f"Expected metrics 'acc=' in status message, got: {msg}"
    win.close()
