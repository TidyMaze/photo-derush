import os
import json
from PIL import Image, ExifTags
from src.cache import ThumbnailCache
import logging

RATINGS_TAGS_PATH = os.path.expanduser('~/.photo-derush-ratings-tags.json')

class ImageModel:
    def __init__(self, directory, max_images=100, cache=None, allowed_exts=None):
        self.directory = directory
        self.max_images = max_images
        self.cache = cache or ThumbnailCache()
        self.allowed_exts = allowed_exts or ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    def set_allowed_exts(self, exts):
        if not isinstance(exts, list) or not all(isinstance(e, str) for e in exts):
            logging.error("allowed_exts must be a list of strings.")
            return
        self.allowed_exts = exts

    def get_image_files(self):
        if not os.path.isdir(self.directory):
            logging.error(f"Directory does not exist: {self.directory}")
            return []
        files = [f for f in os.listdir(self.directory)
                 if os.path.splitext(f)[1].lower() in self.allowed_exts]
        return files[:self.max_images]

    def get_image_path(self, filename):
        if not isinstance(filename, str) or not filename:
            logging.error("Invalid filename provided to get_image_path.")
            return None
        return os.path.join(self.directory, filename)

    def load_exif(self, path):
        try:
            img = Image.open(path)
            exif_data = img._getexif() if hasattr(img, '_getexif') and callable(img._getexif) else None
            if not exif_data or not isinstance(exif_data, dict):
                return {}
            exif = {}
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, str(tag))
                exif[tag_name] = value
            return exif
        except Exception as e:
            logging.warning(f"Failed to load EXIF from {path}: {e}")
            return {}

    def load_thumbnail(self, path, size=128):
        # Check cache first
        thumb = self.cache.get_thumbnail(path)
        if thumb:
            return thumb
        try:
            img = Image.open(path)
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img_cropped = img.crop((left, top, right, bottom)).resize((size, size), Image.Resampling.LANCZOS)
            img_cropped = img_cropped.convert("RGBA")
            self.cache.set_thumbnail(path, img_cropped)
            return img_cropped
        except Exception as e:
            logging.warning(f"Failed to load thumbnail for {path}: {e}")
            return None

    def _load_ratings_tags(self):
        if not hasattr(self, '_ratings_tags'):
            try:
                with open(RATINGS_TAGS_PATH, 'r') as f:
                    self._ratings_tags = json.load(f)
            except Exception as e:
                logging.info(f"No ratings/tags file found or failed to load: {e}")
                self._ratings_tags = {}
        return self._ratings_tags

    def _save_ratings_tags(self):
        try:
            with open(RATINGS_TAGS_PATH, 'w') as f:
                json.dump(self._ratings_tags, f)
        except Exception as e:
            logging.warning(f"Failed to save ratings/tags: {e}")

    def get_rating(self, path):
        self._load_ratings_tags()
        return self._ratings_tags.get(path, {}).get('rating', 0)

    def set_rating(self, path, rating):
        if not isinstance(rating, int) or not (0 <= rating <= 5):
            logging.error("Rating must be an integer between 0 and 5.")
            return
        self._load_ratings_tags()
        if path not in self._ratings_tags:
            self._ratings_tags[path] = {}
        self._ratings_tags[path]['rating'] = rating
        self._save_ratings_tags()

    def get_tags(self, path):
        self._load_ratings_tags()
        return self._ratings_tags.get(path, {}).get('tags', [])

    def set_tags(self, path, tags):
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            logging.error("Tags must be a list of strings.")
            return
        self._load_ratings_tags()
        if path not in self._ratings_tags:
            self._ratings_tags[path] = {}
        self._ratings_tags[path]['tags'] = tags
        self._save_ratings_tags()

    def filter_by_filename(self, substring):
        if not isinstance(substring, str) or not substring:
            return self.get_image_files()
        substring = substring.lower()
        return [f for f in self.get_image_files() if substring in f.lower()]

    def filter_by_exif(self, field, value):
        if not isinstance(field, str) or not field or not isinstance(value, str) or not value:
            return self.get_image_files()
        field = field.lower()
        value = value.lower()
        result = []
        for f in self.get_image_files():
            path = self.get_image_path(f)
            exif = self.load_exif(path)
            for k, v in exif.items():
                if k.lower() == field and value in str(v).lower():
                    result.append(f)
                    break
        return result

    def filter_by_rating_tag_date(self, rating=0, tag='', date=''):
        files = self.get_image_files()
        self._load_ratings_tags()
        filtered = []
        tag = tag.lower().strip() if tag else ''
        date = date.strip() if date else ''
        for f in files:
            path = self.get_image_path(f)
            info = self._ratings_tags.get(path, {})
            # Rating filter
            if rating and info.get('rating', 0) < rating:
                continue
            # Tag filter
            if tag:
                tags = [t.lower() for t in info.get('tags', [])]
                if tag not in tags:
                    continue
            # Date filter (EXIF)
            if date:
                exif = self.load_exif(path)
                exif_date = exif.get('DateTimeOriginal') or exif.get('DateTime') or ''
                if not exif_date.startswith(date):
                    continue
            filtered.append(f)
        return filtered
