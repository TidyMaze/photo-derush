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
    from photo_derush.main_window import compute_feature_vector
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    valid_images = []
    for f in image_paths:
        try:
            fv, keys = compute_feature_vector(os.path.join(image_dir, f))
            if fv is not None:
                valid_images.append(f)
        except Exception:
            continue
        if len(valid_images) >= min_count:
            break
    if len(valid_images) < min_count:
        import pytest
        pytest.skip(f"At least {min_count} valid .jpg images are required in {image_dir}.")
    return image_dir, valid_images

def test_metrics_status_bar():
    app = ensure_app()
    image_dir, image_paths = get_real_image_paths(2)
    img1, img2 = image_paths[:2]
    from photo_derush.main_window import compute_feature_vector, LightroomMainWindow
    fv1, keys1 = compute_feature_vector(os.path.join(image_dir, img1))
    fv2, keys2 = compute_feature_vector(os.path.join(image_dir, img2))
    assert fv1 is not None and fv2 is not None, "Feature extraction failed on real images."
    win = LightroomMainWindow([img1, img2], image_dir, lambda: [img1, img2])
    win.current_img_idx = 0
    win._label_current_image(0)
    win.current_img_idx = 1
    win._label_current_image(1)
    win._refresh_all_keep_probs()
    # Explicitly evaluate model and update status bar
    win._evaluate_model()
    win._update_status_bar()
    for _ in range(20):
        app.processEvents()
    msg = win.status.currentMessage()
    assert msg is None or 'acc=' in msg, f"Expected metrics 'acc=' in status message, got: {msg}"
    win.close()
