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

def test_fullscreen_guard(monkeypatch, tmp_path, caplog):
    app = ensure_app()
    # Create two tiny valid images
    from PIL import Image
    for name in ["a.jpg","b.jpg"]:
        Image.new('RGB',(10,10),color='blue').save(tmp_path/name)
    win = LightroomMainWindow(["a.jpg","b.jpg"], str(tmp_path), lambda: ["a.jpg","b.jpg"])
    win.learner = DummyLearner()
    # Monkeypatch feature_vector to provide simple vectors
    def fv(path):
        import numpy as _np
        # encode prob by name
        base = os.path.basename(path)
        if base == 'a.jpg':
            return _np.array([0.1],dtype=_np.float32), ['p']
        return _np.array([0.9],dtype=_np.float32), ['p']
    monkeypatch.setattr('ml.features.feature_vector', fv)
    monkeypatch.setattr('photo_derush.main_window.feature_vector', fv)
    # Monkeypatch open_full_image_qt to record calls
    calls = []
    def fake_open(path, on_keep=None, on_trash=None, on_unsure=None):
        calls.append(path)
    monkeypatch.setattr('photo_derush.main_window.open_full_image_qt', fake_open)
    # Simulate external erroneous fullscreen calls during sort
    # We'll hook into _sort_by_probability by wrapping open_fullscreen to attempt call
    original_open_fullscreen = win.open_fullscreen
    def noisy_open(idx, path):
        # Force call original (guard should suppress while in sort)
        original_open_fullscreen(idx, path)
    win.open_fullscreen = noisy_open
    # Trigger sort
    win.on_predict_sort_desc()
    # No fullscreen windows should have opened
    assert calls == [], f"Guard failed, fullscreen calls: {calls}"
    # Outside sort it should open
    win._in_sort = False
    win.open_fullscreen(0, os.path.join(str(tmp_path),'a.jpg'))
    assert len(calls) == 1
    for _ in range(5):
        app.processEvents()

