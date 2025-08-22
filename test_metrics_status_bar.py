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
    import tempfile
    from PIL import Image
    from photo_derush.main_window import compute_feature_vector
    import logging
    tmp_dir = tempfile.mkdtemp()
    image_paths = []
    for i in range(min_count):
        img_name = f"test_img_{i}.jpg"
        img_path = os.path.join(tmp_dir, img_name)
        img = Image.new("RGB", (32, 32), color=(255, 0, 0) if i == 0 else (0, 255, 0))
        img.save(img_path)
        fv, keys = compute_feature_vector(img_path)
        logging.warning(f"[DEBUG] compute_feature_vector for {img_path}: fv={fv}, keys={keys}")
        if fv is not None:
            image_paths.append(img_name)
    if len(image_paths) < min_count:
        import pytest
        pytest.skip(f"Could not generate {min_count} valid test images.")
    return tmp_dir, image_paths

def test_metrics_status_bar():
    app = ensure_app()
    image_dir, image_paths = get_real_image_paths(2)
    img1, img2 = image_paths[:2]
    from photo_derush.main_window import compute_feature_vector, LightroomMainWindow
    fv1, keys1 = compute_feature_vector(os.path.join(image_dir, img1))
    fv2, keys2 = compute_feature_vector(os.path.join(image_dir, img2))
    assert fv1 is not None and fv2 is not None, "Feature extraction failed on real images."
    win = LightroomMainWindow([img1, img2], image_dir, lambda: [img1, img2])
    # Explicitly populate the window's feature cache for both images
    mtime1 = os.path.getmtime(os.path.join(image_dir, img1))
    mtime2 = os.path.getmtime(os.path.join(image_dir, img2))
    win._feature_cache[os.path.join(image_dir, img1)] = (mtime1, (fv1, keys1))
    win._feature_cache[os.path.join(image_dir, img2)] = (mtime2, (fv2, keys2))
    win.current_img_idx = 0
    win._label_current_image(0)
    win.current_img_idx = 1
    win._label_current_image(1)
    import logging
    win._refresh_all_keep_probs()
    logging.warning(f'AFTER _refresh_all_keep_probs: {win.status.currentMessage()} {win._last_metrics}')
    assert win._last_metrics is not None, 'No metrics after _refresh_all_keep_probs'
    assert 'acc' in win._last_metrics, f"No 'acc' in metrics after _refresh_all_keep_probs: {win._last_metrics}"
    # Explicitly evaluate model and update status bar
    win._evaluate_model()
    logging.warning(f'AFTER _evaluate_model: {win.status.currentMessage()} {win._last_metrics}')
    assert win._last_metrics is not None, 'No metrics after _evaluate_model'
    assert 'acc' in win._last_metrics, f"No 'acc' in metrics after _evaluate_model: {win._last_metrics}"
    win._update_status_bar()
    logging.warning(f'AFTER _update_status_bar: {win.status.currentMessage()} {win._last_metrics}')
    for _ in range(20):
        app.processEvents()
    msg = win.status.currentMessage()
    logging.warning(f'FINAL STATUS: {msg} {win._last_metrics}')
    assert msg is None or 'acc=' in msg, f"Expected metrics 'acc=' in status message, got: {msg}"
    win.close()
