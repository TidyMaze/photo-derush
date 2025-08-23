import os
from PIL import Image, ExifTags

class ImageModel:
    def __init__(self, directory, max_images=100):
        self.directory = directory
        self.max_images = max_images

    def get_image_files(self):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        files = [f for f in os.listdir(self.directory)
                 if os.path.splitext(f)[1].lower() in image_exts]
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
            return img_cropped
        except Exception:
            return None
