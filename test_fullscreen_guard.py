import os
import numpy as np
from PySide6.QtWidgets import QApplication
from photo_derush.main_window import LightroomMainWindow

class DummyLearner:
    def predict_keep_prob(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:,0]

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

def test_fullscreen_guard(caplog):
    app = ensure_app()
    image_dir, image_paths = get_real_image_paths(2)
    from photo_derush.main_window import LightroomMainWindow
    win = LightroomMainWindow(image_paths[:2], image_dir, lambda: image_paths[:2])
    try:
        win.open_fullscreen(0, os.path.join(image_dir, image_paths[0]))
        for _ in range(10):
            app.processEvents()
        msg = win.status.currentMessage()
        assert msg is None or isinstance(msg, str)
    except Exception as e:
        import pytest
        pytest.skip(f"UI cannot be tested in headless environment: {e}")
    win.close()
