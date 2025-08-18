import os
import numpy as np
from PySide6.QtWidgets import QApplication
from photo_derush.main_window import LightroomMainWindow, _NEW_FEATURE_NAMES

def ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_metrics_status_bar(tmp_path, monkeypatch):
    app = ensure_app()
    # Create two simple images
    from PIL import Image
    for name, color in [("a.jpg", "red"), ("b.jpg", "blue")]:
        Image.new('RGB', (32,32), color=color).save(tmp_path/name)
    # Deterministic tiny feature vector (length must match FEATURE_NAMES)
    vec_len = len(_NEW_FEATURE_NAMES)
    base_vec = np.linspace(0, 1, vec_len).astype(np.float32)
    def fake_compute(path):
        # Slight variation per file
        mod = 0.01 if path.endswith('a.jpg') else -0.01
        return base_vec + mod, list(_NEW_FEATURE_NAMES)
    monkeypatch.setattr('photo_derush.main_window.compute_feature_vector', fake_compute)
    monkeypatch.setattr('ml.features_cv.compute_feature_vector', fake_compute)

    win = LightroomMainWindow(["a.jpg","b.jpg"], str(tmp_path), lambda: ["a.jpg","b.jpg"])  # learner auto-init later
    # Label first image as 0
    win.current_img_idx = 0
    win._label_current_image(0)
    # Label second image as 1
    win.current_img_idx = 1
    win._label_current_image(1)
    # Force probability refresh which triggers evaluation & status message
    win._refresh_all_keep_probs()
    # Process events so status bar updates
    for _ in range(10):
        app.processEvents()
    msg = win.status.currentMessage()
    assert msg is None or 'acc=' in msg, f"Expected metrics 'acc=' in status message, got: {msg}"
    win.close()

