import os
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from photo_derush.main_window import LightroomMainWindow, _NEW_FEATURE_NAMES


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

def test_viewer_auto_advance():
    app = ensure_app()
    image_dir, image_paths = get_real_image_paths(2)
    img1, img2 = image_paths[:2]
    from photo_derush.main_window import compute_feature_vector
    fv1, keys1 = compute_feature_vector(os.path.join(image_dir, img1))
    fv2, keys2 = compute_feature_vector(os.path.join(image_dir, img2))
    assert fv1 is not None and fv2 is not None, "Feature extraction failed on real images."
    from photo_derush import viewer as viewer_mod
    win = LightroomMainWindow([img1, img2], image_dir, lambda: [img1, img2])
    img1_path = os.path.join(image_dir, img1)
    win.open_fullscreen(0, img1_path)
    for _ in range(5):
        app.processEvents()
    viewer = getattr(win, '_embedded_viewer', None)
    assert viewer is not None, "Embedded viewer not created"
    assert viewer.current_index == 0
    viewer.keep_btn.click()
    for _ in range(10):
        app.processEvents()
    assert viewer.current_index == 1, "Viewer did not auto-advance after keep."
    win.close()
