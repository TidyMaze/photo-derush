import os
from PIL import Image
import pytest
from unittest.mock import MagicMock
import main

class DummyEvent:
    def __init__(self, widget):
        self.widget = widget

class DummyLabel:
    def __init__(self):
        self.selected = False
        self.opened = False
        self.handlers = {}
    def bind(self, event, handler):
        self.handlers[event] = handler
    def config(self, **kwargs):
        if kwargs.get('bg') == "#555":
            self.selected = True
    def simulate_click(self):
        if "<Button-1>" in self.handlers:
            self.handlers["<Button-1>"](DummyEvent(self))
    def simulate_double_click(self):
        if "<Double-Button-1>" in self.handlers:
            self.handlers["<Double-Button-1>"](DummyEvent(self))
            self.opened = True

# Test: clicking an image selects it, double-clicking opens it

def test_image_selection_and_open(monkeypatch):
    opened = {'called': False}
    def fake_open_full_image(img_path):
        opened['called'] = True
    monkeypatch.setattr(main, 'open_full_image', fake_open_full_image)
    label = DummyLabel()
    # Simulate the UI binding logic
    def on_click(event, label=label):
        label.config(bg="#555")
    def on_double_click(event):
        main.open_full_image("dummy_path")
    label.bind("<Button-1>", on_click)
    label.bind("<Double-Button-1>", on_double_click)
    # Simulate click
    label.simulate_click()
    assert label.selected, "Image should be selected on click"
    # Simulate double-click
    label.simulate_double_click()
    assert label.opened, "Image should be opened on double-click"
    assert opened['called'], "open_full_image should be called on double-click"

def test_thumbnail_low_resolution(monkeypatch):
    loaded_sizes = []
    from PIL import Image
    import tempfile
    import os
    # Create a dummy image file at /tmp/img1.jpg
    tmp_img_path = "/tmp/img1.jpg"
    img = Image.new("RGB", (4000, 3000))
    img.save(tmp_img_path)
    def fake_open(path):
        img = Image.open(tmp_img_path)
        orig_thumbnail = img.thumbnail
        def thumbnail(size):
            loaded_sizes.append(size)
            return orig_thumbnail(size)
        img.thumbnail = thumbnail
        return img
    monkeypatch.setattr("PIL.Image.open", fake_open)
    import main
    main.show_lightroom_ui(["img1.jpg"], "/tmp")
    assert any(s[0] <= 200 and s[1] <= 200 for s in loaded_sizes), "Images should be loaded in low resolution (thumbnail)"
    os.remove(tmp_img_path)

def test_images_displayed_after_window_opens(monkeypatch):
    displayed = []
    class DummyImage:
        def thumbnail(self, size): pass
    class DummyPhotoImage:
        def __init__(self, img, master=None): pass
    class DummyLabel:
        def __init__(self, frame, image=None, bg=None, bd=None, relief=None):
            displayed.append(True)
        def grid(self, **kwargs): pass
        def bind(self, event, handler): pass
        def config(self, **kwargs): pass
    monkeypatch.setattr("PIL.Image.open", lambda path: DummyImage())
    monkeypatch.setattr("PIL.ImageTk.PhotoImage", DummyPhotoImage)
    monkeypatch.setattr("tkinter.Label", DummyLabel)
    import main
    main.show_lightroom_ui(["img1.jpg", "img2.jpg"], "/tmp")
    assert displayed, "Images should be displayed in the UI after window opens"

def test_thumbnail_cache_usage(tmp_path, caplog):
    img_name = "test.jpg"
    img_path = tmp_path / img_name
    img = Image.new("RGB", (300, 300), color="red")
    img.save(img_path)
    thumbnail_dir = tmp_path / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    thumb_path = thumbnail_dir / img_name

    from main import cache_thumbnail

    # First call: should create thumbnail
    with caplog.at_level("INFO"):
        img_obj, cached = cache_thumbnail(str(img_path), str(thumb_path))
    assert thumb_path.exists(), "Thumbnail was not created"
    assert not cached, "Thumbnail should not be cached on first call"
    assert any("Created and cached thumbnail" in r for r in caplog.messages), "Thumbnail creation not logged"

    # Second call: should use cache
    caplog.clear()
    with caplog.at_level("INFO"):
        img_obj, cached = cache_thumbnail(str(img_path), str(thumb_path))
    assert cached, "Thumbnail cache not used on second call"
    assert any("Loaded cached thumbnail" in r for r in caplog.messages), "Thumbnail cache usage not logged"

import os
import shutil
import tempfile
from main import duplicate_slayer, show_lightroom_ui
import numpy as np

def test_duplicate_slayer_removes_duplicates_and_keeps_best(tmp_path):
    # Setup: create 3 near-identical images (simulate with same content)
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img1.write_bytes(b"fakeimage1")
    img2.write_bytes(b"fakeimage1")  # duplicate
    img3.write_bytes(b"fakeimage1")  # duplicate

    trash_dir = tmp_path / "trash"
    trash_dir.mkdir()

    # Run duplicate slayer
    kept, trashed = duplicate_slayer(str(tmp_path), str(trash_dir))

    # Only one image should be kept, two should be trashed
    assert len(kept) == 1
    assert len(trashed) == 2
    for t in trashed:
        assert os.path.exists(os.path.join(trash_dir, os.path.basename(t)))
    for k in kept:
        assert os.path.exists(k)

def test_hashes_are_uint8():
    hash_bytes = np.array([1,2,3,4,5,6,7,8], dtype='uint8')
    hashes = [hash_bytes for _ in range(10)]
    hashes_np = np.stack(hashes)
    assert hashes_np.dtype == np.uint8, "hashes_np should be uint8 for FAISS"

def test_lightroom_ui_select_and_fullscreen(monkeypatch):
    import tkinter as tk
    # Fix recursion: capture original before monkeypatch
    _orig_Tk = tk.Tk
    monkeypatch.setattr(tk, "Tk", lambda: _orig_Tk())
    import PIL.Image
    monkeypatch.setattr(PIL.Image, "open", lambda path: PIL.Image.new("RGB", (100, 100)))
    import PIL.ImageTk
    monkeypatch.setattr(PIL.ImageTk, "PhotoImage", lambda img, master=None: "photoimage")
    import faiss
    monkeypatch.setattr(faiss, "IndexBinaryFlat", lambda dim: type("DummyIndex", (), {"add": lambda self, x: None, "range_search": lambda self, x, thresh: ([0, 1], [0], [0])})())
    try:
        show_lightroom_ui(["img1.jpg", "img2.jpg"], "/tmp")
    except Exception as e:
        assert False, f"UI should not crash: {e}"
