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

def test_viewer_auto_advance(tmp_path, monkeypatch):
    app = ensure_app()
    # Create two tiny images
    from PIL import Image
    for name in ["a.jpg","b.jpg"]:
        Image.new('RGB',(20,20),color='green').save(tmp_path/name)
    # Monkeypatch feature extraction to fast deterministic vector
    def fake_fv(path):
        # Return vector length matching FEATURE names
        return np.zeros(len(_NEW_FEATURE_NAMES), dtype=np.float32), list(_NEW_FEATURE_NAMES)
    monkeypatch.setattr('photo_derush.main_window.compute_feature_vector', fake_fv)
    from photo_derush import viewer as viewer_mod
    # Also patch in viewer (embedded viewer loads images but labeling uses main_window compute)
    monkeypatch.setattr('ml.features_cv.compute_feature_vector', fake_fv)

    win = LightroomMainWindow(["a.jpg","b.jpg"], str(tmp_path), lambda: ["a.jpg","b.jpg"])
    # Open embedded fullscreen
    img_a_path = os.path.join(str(tmp_path), 'a.jpg')
    win.open_fullscreen(0, img_a_path)
    for _ in range(5):
        app.processEvents()
    viewer = getattr(win, '_embedded_viewer', None)
    assert viewer is not None, "Embedded viewer not created"
    assert viewer.current_index == 0
    # Click keep button -> should label and auto-advance
    viewer.keep_btn.click()
    for _ in range(10):
        app.processEvents()
    assert viewer.current_index == 1, "Viewer did not auto-advance after labeling"
    assert win.current_img_idx == 1, "Main window index not updated after auto-advance"
    # Clean up
    win._restore_from_viewer()
    win.close()
    for _ in range(3):
        app.processEvents()

