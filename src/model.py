import os
from PIL import Image, ExifTags
from cache import ThumbnailCache

class ImageModel:
    def __init__(self, directory, max_images=100, cache=None, allowed_exts=None):
        self.directory = directory
        self.max_images = max_images
        self.cache = cache or ThumbnailCache()
        self.allowed_exts = allowed_exts or ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    def set_allowed_exts(self, exts):
        self.allowed_exts = exts

    def get_image_files(self):
        files = [f for f in os.listdir(self.directory)
                 if os.path.splitext(f)[1].lower() in self.allowed_exts]
        return files[:self.max_images]

    def get_image_path(self, filename):
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
        except Exception:
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
        except Exception:
            return None
