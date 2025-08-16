import os
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PIL import Image

from photo_derush.main_window import LightroomMainWindow

class DummyLearner:
    def predict_keep_prob(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Probability encoded in first feature
        return arr[:, 0]

def _ensure_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

def test_probability_sort_asc_desc(tmp_path):
    app = _ensure_qapp()
    prob_map = {'imgA.jpg': 0.9, 'imgB.jpg': 0.2, 'imgC.jpg': 0.5}
    # Create valid tiny images
    for name in prob_map.keys():
        img = Image.new('RGB', (10,10), color='red')
        img.save(tmp_path / name)
    win = LightroomMainWindow([], str(tmp_path), lambda: [])
    win.sorted_images = ['imgB.jpg','imgC.jpg','imgA.jpg']
    win.learner = DummyLearner()
    # Seed feature cache: path -> (mtime, (fv, keys)) with first element = desired prob
    for name, p in prob_map.items():
        path = os.path.join(str(tmp_path), name)
        mtime = os.path.getmtime(path)
        fv = np.array([p, 0.0, 0.0], dtype=np.float32)
        win._feature_cache[path] = (mtime, (fv, ['p','f1','f2']))
    win.on_predict_sort_desc()
    assert win.sorted_images == ['imgA.jpg','imgC.jpg','imgB.jpg'], f"Desc sort wrong: {win.sorted_images}"
    win.on_predict_sort_asc()
    assert win.sorted_images == ['imgB.jpg','imgC.jpg','imgA.jpg'], f"Asc sort wrong: {win.sorted_images}"
    for _ in range(5):
        app.processEvents()
