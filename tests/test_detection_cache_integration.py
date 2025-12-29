import joblib
import os

import pytest

from src.viewmodel import PhotoViewModel
import src.object_detection as od


def test_viewmodel_reads_cache_and_populates_detected_objects(tmp_path, monkeypatch):
    # Create a temp dir and image file
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    img_file = img_dir / 'photo.jpg'
    img_file.write_bytes(b'')  # empty file; VM only uses basename

    # Prepare cache file in tmp and monkeypatch get_cache_path
    cache_file = tmp_path / 'object_detections.joblib'
    monkeypatch.setattr(od, 'get_cache_path', lambda: str(cache_file))

    # Sample detections (already canonical shape)
    sample = {'photo.jpg': [{'class': 'cat', 'confidence': 0.85, 'bbox': [1, 2, 3, 4]}]}
    cache_data = {'detections': sample, 'backend': od.DETECTION_BACKEND}
    os.makedirs(os.path.dirname(str(cache_file)), exist_ok=True)
    joblib.dump(cache_data, str(cache_file))

    # Initialize VM pointing to the directory
    vm = PhotoViewModel(str(img_dir))
    vm.images = ['photo.jpg']

    # Force the batch API to return nothing so the ViewModel uses the
    # cache entries directly (preserving bbox coordinates).
    # Return None (not dict) so the synchronous batch-path is skipped
    monkeypatch.setattr(od, 'get_objects_for_images', lambda paths: None)

    # Load detections (synchronous path should read cache)
    vm._load_object_detections()

    # Assert the VM preserved the bbox from cache (no fallback stripping)
    assert 'photo.jpg' in vm._detected_objects
    assert vm._detected_objects['photo.jpg'] == sample['photo.jpg']
