import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.model import ImageModel

def test_exif_disk_cache_persistence():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock cache file path inside temp_dir
        cache_file = os.path.join(temp_dir, "exif_cache.pkl")
        
        # Instantiate ImageModel
        model = ImageModel(temp_dir)
        # Force cache path to be inside temp_dir
        model._exif_cache_path = cache_file
        
        # Populate the cache manually
        dummy_path = os.path.join(temp_dir, "dummy.jpg")
        dummy_exif = {"Make": "Sony", "Model": "ILCE-7M3"}
        model._exif_cache[dummy_path] = dummy_exif
        
        # Save cache
        model.save_exif_cache()
        assert os.path.exists(cache_file)
        
        # Create a new model instance and verify it loads the cache
        model2 = ImageModel(temp_dir)
        model2._exif_cache_path = cache_file
        model2._exif_cache = model2._load_exif_cache()
        
        assert dummy_path in model2._exif_cache
        assert model2._exif_cache[dummy_path] == dummy_exif
