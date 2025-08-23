import os
import tempfile
import shutil
import pytest
import sys

sys.path = [p for p in sys.path if 'src' not in p]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import ImageModel

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

def test_filter_by_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'foo.jpg'), 'a').close()
        open(os.path.join(tmpdir, 'bar.JPG'), 'a').close()
        open(os.path.join(tmpdir, 'baz.png'), 'a').close()
        model = ImageModel(tmpdir)
        assert set(model.filter_by_filename('foo')) == {'foo.jpg'}
        assert set(model.filter_by_filename('BAZ')) == {'baz.png'}
        assert set(model.filter_by_filename('jpg')) == {'foo.jpg', 'bar.JPG'}
        assert set(model.filter_by_filename('nope')) == set()
        # Empty/None returns all
        all_files = set(model.get_image_files())
        assert set(model.filter_by_filename('')) == all_files
        assert set(model.filter_by_filename(None)) == all_files

def test_filter_by_exif(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, 'a.jpg'), 'a').close()
        open(os.path.join(tmpdir, 'b.jpg'), 'a').close()
        model = ImageModel(tmpdir)
        # Patch load_exif to return fake data
        def fake_load_exif(path):
            if path.endswith('a.jpg'):
                return {'Model': 'Canon', 'DateTime': '2020:01:01'}
            if path.endswith('b.jpg'):
                return {'Model': 'Nikon', 'DateTime': '2021:01:01'}
            return {}
        monkeypatch.setattr(model, 'load_exif', fake_load_exif)
        assert set(model.filter_by_exif('Model', 'Canon')) == {'a.jpg'}
        assert set(model.filter_by_exif('model', 'nikon')) == {'b.jpg'}
        assert set(model.filter_by_exif('DateTime', '2020')) == {'a.jpg'}
        assert set(model.filter_by_exif('DateTime', '2022')) == set()
        # Empty/None returns all
        all_files = set(model.get_image_files())
        assert set(model.filter_by_exif('', 'Canon')) == all_files
        assert set(model.filter_by_exif('Model', '')) == all_files
        assert set(model.filter_by_exif(None, None)) == all_files
