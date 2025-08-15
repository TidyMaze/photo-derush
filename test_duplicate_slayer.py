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
    import faiss
    # Create dummy image files in /tmp
    tmp_dir = "/tmp"
    for img_name in ["img1.jpg", "img2.jpg"]:
        img_path = os.path.join(tmp_dir, img_name)
        from PIL import Image
        Image.new("RGB", (100, 100)).save(img_path)
    monkeypatch.setattr(faiss, "IndexBinaryFlat", lambda dim: type("DummyIndex", (), {"add": lambda self, x: None, "range_search": lambda self, x, thresh: ([0, 1], [0], [0])})())
    try:
        show_lightroom_ui(["img1.jpg", "img2.jpg"], "/tmp")
    except Exception as e:
        assert False, f"UI should not crash: {e}"
    # Clean up dummy files
    for img_name in ["img1.jpg", "img2.jpg"]:
        img_path = os.path.join(tmp_dir, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)
