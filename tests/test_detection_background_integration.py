import joblib
import os

import pytest

from src.viewmodel import PhotoViewModel
import src.object_detection as od


def test_background_detection_populates_cache_with_bbox(tmp_path, monkeypatch):
    # Setup image dir and file
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    (img_dir / 'photo.jpg').write_bytes(b'')

    # Monkeypatch cache path to tmp
    cache_file = tmp_path / 'object_detections.joblib'
    monkeypatch.setattr(od, 'get_cache_path', lambda: str(cache_file))

    # Ensure synchronous batch path returns nothing so background path runs
    monkeypatch.setattr(od, 'get_objects_for_images', lambda paths: None)

    # Patch detect_objects to return a bbox-containing detection
    sample_det = {'class': 'dog', 'confidence': 0.77, 'bbox': [5, 6, 7, 8]}
    monkeypatch.setattr(od, 'detect_objects', lambda path, *args, **kwargs: [sample_det])

    # Initialize VM and replace TaskRunner.run to execute inline
    vm = PhotoViewModel(str(img_dir))
    vm.images = ['photo.jpg']
    monkeypatch.setattr(vm._tasks, 'run', lambda name, fn: fn(None))

    # Run loader which will schedule and execute the background detect task inline
    vm._load_object_detections()

    # After background task, VM should have bbox in detected_objects and cache should be written
    assert 'photo.jpg' in vm._detected_objects
    assert vm._detected_objects['photo.jpg'] == [sample_det]

    # Check on-disk cache contains bbox
    loaded = od.load_object_cache()
    assert 'photo.jpg' in loaded
    assert loaded['photo.jpg'] == [sample_det]
