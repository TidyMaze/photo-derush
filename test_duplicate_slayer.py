import os
import shutil
import tempfile
from main import duplicate_slayer

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

