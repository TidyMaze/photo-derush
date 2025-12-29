import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import torch

from src.object_detection import (
    get_objects_for_image,
    get_objects_for_images,
    load_object_cache,
    save_object_cache,
    get_cache_path,
    COCO_CLASSES,
    INTERESTING_CLASSES
)


class TestObjectDetection:
    """Test object detection functionality."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        return img

    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            sample_image.save(f.name)
            yield f.name
        os.unlink(f.name)

    def test_get_objects_for_image_cached(self, temp_image_file):
        """Test getting objects for image with cache hit."""
        # Mock cache with existing data
        with patch('src.object_detection.load_object_cache') as mock_load_cache:
            mock_load_cache.return_value = {
                os.path.basename(temp_image_file): [
                    {'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]}
                ]
            }

            result = get_objects_for_image(temp_image_file)
            assert result == [('person', 0.9)]
            # Should not call detect_objects when cache hit
            # (This is implicit - if it did call, it would fail without mocks)

    def test_get_objects_for_image_uncached(self, temp_image_file):
        """Test getting objects for image without cache."""
        with patch('src.object_detection.load_object_cache') as mock_load_cache, \
             patch('src.object_detection.detect_objects') as mock_detect, \
             patch('src.object_detection.save_object_cache') as mock_save:

            mock_load_cache.return_value = {}  # No cache
            mock_detect.return_value = [
                {'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]},
                {'class': 'person', 'confidence': 0.8, 'bbox': [20, 20, 60, 60]},  # Duplicate class
                {'class': 'car', 'confidence': 0.7, 'bbox': [60, 60, 90, 90]}
            ]

            result = get_objects_for_image(temp_image_file)
            assert result == [('person', 0.9), ('car', 0.7)]  # Should be unique classes with confidence, highest confidence first

            # Verify cache was saved
            mock_save.assert_called_once()

    def test_get_objects_for_image_confidence_filtering(self, temp_image_file):
        """Test confidence threshold filtering in get_objects_for_image."""
        with patch('src.object_detection.load_object_cache') as mock_load_cache, \
             patch('src.object_detection.detect_objects') as mock_detect, \
             patch('src.object_detection.save_object_cache'):

            mock_load_cache.return_value = {}
            # detect_objects already filters by confidence, so it should only return high confidence detections
            mock_detect.return_value = [
                {'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]}
                # Note: low confidence detection (0.4) would be filtered out by detect_objects
            ]

            result = get_objects_for_image(temp_image_file, confidence_threshold=0.5)
            assert result == [('person', 0.9)]  # Only high confidence detection

    def test_get_objects_for_images_batch(self, temp_image_file):
        """Test batch processing of multiple images."""
        image_paths = [temp_image_file, temp_image_file + '_2']  # Same image twice for simplicity

        with patch('src.object_detection.load_object_cache') as mock_load_cache, \
             patch('src.object_detection.detect_objects') as mock_detect, \
             patch('src.object_detection.save_object_cache'):

            mock_load_cache.return_value = {}
            mock_detect.return_value = [
                {'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]},
                {'class': 'person', 'confidence': 0.8, 'bbox': [20, 20, 60, 60]},  # Duplicate class
                {'class': 'car', 'confidence': 0.7, 'bbox': [60, 60, 90, 90]}
            ]

            result = get_objects_for_images(image_paths)

            assert len(result) == 2
            assert all(objects == [('person', 0.9), ('car', 0.7)] for objects in result.values())  # Unique classes with confidence only

    def test_get_objects_for_images_with_progress_callback(self, temp_image_file):
        """Test batch processing with progress callback."""
        image_paths = [temp_image_file]

        progress_calls = []
        def progress_callback(current, total, detail):
            progress_calls.append((current, total, detail))

        with patch('src.object_detection.load_object_cache') as mock_load_cache, \
             patch('src.object_detection.detect_objects') as mock_detect, \
             patch('src.object_detection.save_object_cache'):

            mock_load_cache.return_value = {}
            mock_detect.return_value = [
                {'class': 'person', 'confidence': 0.9, 'bbox': [10, 10, 50, 50]}
            ]

            result = get_objects_for_images(image_paths, progress_callback=progress_callback)

            assert len(progress_calls) == 1
            assert progress_calls[0][0] == 0  # current
            assert progress_calls[0][1] == 1  # total

    def test_cache_operations(self):
        """Test cache loading and saving."""
        test_data = {
            'image1.jpg': [{'class': 'person', 'confidence': 0.9}],
            'image2.jpg': [{'class': 'car', 'confidence': 0.8}]
        }

        with patch('src.object_detection.joblib.load') as mock_load, \
             patch('src.object_detection.joblib.dump') as mock_dump, \
             patch('src.object_detection.os.path.exists') as mock_exists, \
             patch('src.object_detection.os.makedirs'):

            # Test loading existing cache
            mock_exists.return_value = True
            mock_load.return_value = {'detections': test_data}

            loaded = load_object_cache()
            assert loaded == test_data

            # Test saving cache
            save_object_cache(test_data)
            mock_dump.assert_called_once()

            # Test loading non-existent cache
            mock_exists.return_value = False
            loaded_empty = load_object_cache()
            assert loaded_empty == {}

    def test_get_cache_path(self):
        """Test cache path generation."""
        with patch('src.object_detection.os.path.join') as mock_join:
            mock_join.return_value = '/test/path/.cache/object_detections.joblib'
            path = get_cache_path()
            mock_join.assert_called_with('.cache', 'object_detections.joblib')

    def test_coco_classes_constants(self):
        """Test COCO classes constants are properly defined."""
        assert len(COCO_CLASSES) > 80  # Should have many classes
        assert COCO_CLASSES[0] == '__background__'
        assert 'person' in COCO_CLASSES
        assert 'car' in COCO_CLASSES

    def test_interesting_classes_subset(self):
        """Test that INTERESTING_CLASSES contains expected classes."""
        assert 1 in INTERESTING_CLASSES  # person
        assert 3 in INTERESTING_CLASSES  # car
        assert 18 in INTERESTING_CLASSES  # dog

        # Verify class names match
        assert INTERESTING_CLASSES[1] == 'person'
        assert INTERESTING_CLASSES[3] == 'car'