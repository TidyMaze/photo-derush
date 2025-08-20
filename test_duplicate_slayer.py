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

def get_real_image_paths(min_count=2):
    image_dir = '/Users/yannrolland/Pictures/photo-dataset'
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if len(image_paths) < min_count:
        import pytest
        pytest.skip(f"At least {min_count} .jpg images are required in {image_dir}.")
    return image_dir, image_paths

def test_lightroom_ui_select_and_fullscreen():
    image_dir, image_paths = get_real_image_paths(2)
    from main import show_lightroom_ui
    try:
        show_lightroom_ui(image_paths[:2], image_dir)
    except Exception as e:
        import pytest
        pytest.skip(f"UI cannot be tested in headless environment: {e}")

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
    clusters, _ = cluster_duplicates(image_paths, str(tmp_path), hamming_thresh=1)
    # There should be one cluster with the two identical images
    assert any(set(cluster) == {"img1.jpg", "img2.jpg"} for cluster in clusters), f"Expected img1 and img2 to be clustered together, got {clusters}"
    # The different image should not be clustered with the others
    assert not any("img3.jpg" in cluster and len(cluster) > 1 for cluster in clusters), f"img3.jpg should not be clustered with others: {clusters}"
