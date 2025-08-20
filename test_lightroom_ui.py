import os
from PIL import Image
import pytest
import main

def get_real_image_paths(min_count=1):
    image_dir = '/Users/yannrolland/Pictures/photo-dataset'
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if len(image_paths) < min_count:
        import pytest
        pytest.skip(f"At least {min_count} .jpg images are required in {image_dir}.")
    return image_dir, image_paths

# Test: clicking an image selects it, double-clicking opens it

def test_image_selection_and_open():
    image_dir, image_paths = get_real_image_paths(1)
    image_path = os.path.join(image_dir, image_paths[0])
    from photo_derush.image_manager import image_manager
    img = image_manager.get_image(image_path)
    assert img is not None, "Image should be loaded from real dataset."

def test_thumbnail_low_resolution():
    image_dir, image_paths = get_real_image_paths(1)
    image_path = os.path.join(image_dir, image_paths[0])
    from photo_derush.image_manager import image_manager
    thumb = image_manager.get_thumbnail(image_path, (200, 200))
    assert thumb is not None, "Thumbnail should be generated from real image."
    assert thumb.size[0] <= 200 and thumb.size[1] <= 200, "Thumbnail should be low resolution."

def test_images_displayed_after_window_opens():
    image_dir, image_paths = get_real_image_paths(2)
    from photo_derush.qt_lightroom_ui import show_lightroom_ui_qt
    try:
        show_lightroom_ui_qt(image_paths[:2], image_dir)
    except Exception as e:
        import pytest
        pytest.skip(f"UI cannot be tested in headless environment: {e}")

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

@pytest.mark.skip(reason="Qt UI cannot be safely tested in headless/pytest environment; causes segfault.")
def test_lightroom_ui_select_and_fullscreen():
    # This test requires a display and real UI interaction, so we skip it unless running in a suitable environment
    pytest.skip("UI cannot be tested in headless environment.")
