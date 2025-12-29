import joblib
import os

import pytest

from src import object_detection as od


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    # Redirect cache path to tmp to avoid touching repo cache
    cache_file = tmp_path / 'object_detections.joblib'
    monkeypatch.setattr(od, 'get_cache_path', lambda: str(cache_file))

    sample = {
        'img1.jpg': [
            {'class': 'cat', 'confidence': 0.9, 'bbox': [1, 2, 3, 4]},
        ],
        'img2.png': [],
    }

    od.save_object_cache(sample)

    loaded = od.load_object_cache()
    assert loaded == sample


def test_get_objects_for_images_reads_cache_and_returns_unique_classes(tmp_path, monkeypatch):
    cache_file = tmp_path / 'object_detections.joblib'
    monkeypatch.setattr(od, 'get_cache_path', lambda: str(cache_file))

    sample = {
        'photo.jpg': [
            {'class': 'dog', 'confidence': 0.8, 'bbox': [0, 0, 10, 10]},
            {'class': 'dog', 'confidence': 0.6, 'bbox': [1, 1, 2, 2]},
            {'class': 'cat', 'confidence': 0.7, 'bbox': None},
        ]
    }

    # write cache directly using joblib format expected by load_object_cache
    cache_data = {'detections': sample, 'backend': od.DETECTION_BACKEND}
    os.makedirs(os.path.dirname(str(cache_file)), exist_ok=True)
    joblib.dump(cache_data, str(cache_file))

    # Use a path whose basename matches the cache key
    results = od.get_objects_for_images([str(tmp_path / 'photo.jpg')])
    assert 'photo.jpg' in results
    assert results['photo.jpg'] == [('dog', 0.8), ('cat', 0.7)]


def test_load_invalid_backend_invalidates_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / 'object_detections.joblib'
    monkeypatch.setattr(od, 'get_cache_path', lambda: str(cache_file))

    sample = {'a.jpg': [{'class': 'bird', 'confidence': 0.5, 'bbox': None}]}
    # write a cache with a different backend value to simulate stale cache
    cache_data = {'detections': sample, 'backend': 'not_the_current_backend'}
    os.makedirs(os.path.dirname(str(cache_file)), exist_ok=True)
    joblib.dump(cache_data, str(cache_file))

    loaded = od.load_object_cache()
    assert loaded == {}
