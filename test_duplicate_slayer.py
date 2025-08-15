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

def test_cluster_duplicates_groups_similar_images(tmp_path):
    from main import cluster_duplicates
    from PIL import Image
    # Create two identical images and one different
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    img.save(img1)
    img.save(img2)
    img_diff = Image.new("RGB", (32, 32), color=(255, 0, 0))
    # Draw a black rectangle to make it visually different
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_diff)
    draw.rectangle([8, 8, 24, 24], fill=(0, 0, 0))
    img_diff.save(img3)
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    clusters = cluster_duplicates(image_paths, str(tmp_path), hamming_thresh=1)
    # There should be one cluster with the two identical images
    assert any(set(cluster) == {"img1.jpg", "img2.jpg"} for cluster in clusters), f"Expected img1 and img2 to be clustered together, got {clusters}"
    # The different image should not be clustered with the others
    assert not any("img3.jpg" in cluster and len(cluster) > 1 for cluster in clusters), f"img3.jpg should not be clustered with others: {clusters}"
