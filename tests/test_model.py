import os
import tempfile
import shutil
import pytest
from src.model import ImageModel

class DummyCache:
    def __init__(self):
        self.thumbs = {}
    def get_thumbnail(self, path):
        return self.thumbs.get(path)
    def set_thumbnail(self, path, img):
        self.thumbs[path] = img

def test_get_image_files_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = ImageModel(tmpdir)
        assert model.get_image_files() == []

def test_get_image_files_with_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'a.jpg'), 'a').close()
        open(os.path.join(tmpdir, 'b.png'), 'a').close()
        open(os.path.join(tmpdir, 'c.txt'), 'a').close()
        model = ImageModel(tmpdir)
        files = model.get_image_files()
        assert set(files) == {'a.jpg', 'b.png'}

def test_get_image_path_invalid():
    model = ImageModel('.')
    assert model.get_image_path('') is None
    assert model.get_image_path(None) is None

def test_set_allowed_exts_invalid():
    model = ImageModel('.')
    model.set_allowed_exts('jpg')
    assert model.allowed_exts == ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    model.set_allowed_exts([1, 2])
    assert model.allowed_exts == ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

def test_set_rating_invalid():
    model = ImageModel('.')
    model.set_rating('foo', 10)  # Should not set
    assert model.get_rating('foo') == 0
    model.set_rating('foo', 'bad')
    assert model.get_rating('foo') == 0

def test_set_tags_invalid():
    model = ImageModel('.')
    model.set_tags('foo', 'bar')
    assert model.get_tags('foo') == []
    model.set_tags('foo', [1, 2])
    assert model.get_tags('foo') == []

def test_thumbnail_cache():
    model = ImageModel('.', cache=DummyCache())
    # Should not raise, but returns None since file doesn't exist
    assert model.load_thumbnail('notfound.jpg') is None

def test_load_exif_invalid():
    model = ImageModel('.')
    assert model.load_exif('notfound.jpg') == {}

def test_ratings_tags_persistence(tmp_path):
    model = ImageModel('.')
    path = str(tmp_path / 'img.jpg')
    model.set_rating(path, 3)
    assert model.get_rating(path) == 3
    model.set_tags(path, ['foo', 'bar'])
    assert model.get_tags(path) == ['foo', 'bar']

