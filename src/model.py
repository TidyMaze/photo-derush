import logging
import os

from PIL import ExifTags, Image

from .cache import ThumbnailCache
from .repository import RatingsTagsRepository


class ImageModel:
    def __init__(
        self,
        directory,
        max_images=100,
        cache=None,
        allowed_exts=None,
        repo: RatingsTagsRepository | None = None,
        filtering_service=None,
    ):
        self.directory = directory
        self.max_images = max_images
        self.cache = cache or ThumbnailCache()
        self.allowed_exts = allowed_exts or [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
        self._repo = repo or RatingsTagsRepository()

    def set_allowed_exts(self, exts):
        is_valid_list = isinstance(exts, list) and all(isinstance(e, str) for e in exts)
        if not is_valid_list:
            logging.error("allowed_exts must be a list of strings.")
            return
        self.allowed_exts = exts

    def get_image_files(self):
        logging.info(f"Scanning directory: {self.directory}")
        if not os.path.isdir(self.directory):
            logging.error(f"Directory does not exist: {self.directory}")
            return []
        files = [f for f in os.listdir(self.directory) if os.path.splitext(f)[1].lower() in self.allowed_exts]
        logging.info(f"Found files: {files[:5]}... (total {len(files)})")
        # Unlimited if max_images is None or <= 0
        if self.max_images is None or (isinstance(self.max_images, int) and self.max_images <= 0):
            return files
        return files[: self.max_images]

    def get_image_path(self, filename):
        # PERFORMANCE: Cache path lookups (21,861 calls -> significant savings)
        if not hasattr(self, "_image_path_cache"):
            self._image_path_cache = {}
        
        if filename in self._image_path_cache:
            return self._image_path_cache[filename]
        
        is_valid_filename = isinstance(filename, str) and filename
        if not is_valid_filename:
            logging.error("Invalid filename provided to get_image_path.")
            return None
        
        path = os.path.join(self.directory, filename)
        self._image_path_cache[filename] = path
        return path

    def _filename_from_path(self, path):
        """Extract filename from path for repository lookups."""
        return os.path.basename(path)

    def load_exif(self, path):
        try:
            img = Image.open(path)
            exif_data = img._getexif() if hasattr(img, "_getexif") and callable(img._getexif) else None
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
            # Preserve aspect ratio: use thumbnail() instead of square crop + resize
            img_thumb = img.copy().convert("RGBA")
            img_thumb.thumbnail((size, size), resample=Image.Resampling.LANCZOS)

            # Center on square canvas to maintain consistent thumbnail grid
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            offset_x = (size - img_thumb.width) // 2
            offset_y = (size - img_thumb.height) // 2
            canvas.paste(img_thumb, (offset_x, offset_y))

            # Store metadata directly in image.info for bbox denormalization
            canvas.info["thumb_width"] = str(img_thumb.width)
            canvas.info["thumb_height"] = str(img_thumb.height)
            canvas.info["thumb_offset_x"] = str(offset_x)
            canvas.info["thumb_offset_y"] = str(offset_y)

            logging.debug(
                f"Created thumbnail for {path}: {img_thumb.width}x{img_thumb.height}, offset ({offset_x},{offset_y})"
            )
            self.cache.set_thumbnail(path, canvas)
            return canvas
        except Exception as e:
            logging.warning(f"Failed to load thumbnail for {path}: {e}")
            return None

    # Ratings / Tags / State API delegates to repository
    def get_rating(self, path):
        filename = self._filename_from_path(path)
        rating = self._repo.get_rating(filename)
        logging.debug(f"get_rating: path={path}, key={filename}, rating={rating}")
        return rating

    def set_rating(self, path, rating):
        if not isinstance(rating, int) or not (0 <= rating <= 5):
            logging.error("Rating must be an integer between 0 and 5.")
            return
        filename = self._filename_from_path(path)
        self._repo.set_rating(filename, rating)
        logging.info(f"set_rating: path={path}, key={filename}, rating={rating}")

    def get_tags(self, path):
        filename = self._filename_from_path(path)
        return self._repo.get_tags(filename)

    def set_tags(self, path, tags):
        is_valid_tags = isinstance(tags, list) and all(isinstance(tag, str) for tag in tags)
        if not is_valid_tags:
            logging.error("Tags must be a list of strings.")
            return
        filename = self._filename_from_path(path)
        self._repo.set_tags(filename, tags)

    def get_state(self, path):
        filename = self._filename_from_path(path)
        return self._repo.get_state(filename)

    def set_state(self, path, state, source="manual"):
        filename = self._filename_from_path(path)
        self._repo.set_state(filename, state, source)

    def filter_by_filename(self, substring):
        if not isinstance(substring, str) or not substring:
            return self.get_image_files()
        substring = substring.lower()
        return [f for f in self.get_image_files() if substring in f.lower()]

    def filter_by_exif(self, field, value):
        is_valid = isinstance(field, str) and field and isinstance(value, str) and value
        if not is_valid:
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


    def get_objects(self, path):
        filename = self._filename_from_path(path)
        return self._repo.get_objects(filename)

    def set_objects(self, path, objects):
        is_valid_objects = isinstance(objects, list) and all(isinstance(obj, str) for obj in objects)
        if not is_valid_objects:
            logging.error("Objects must be a list of strings.")
            return
        filename = self._filename_from_path(path)
        self._repo.set_objects(filename, objects)
        logging.info(f"set_objects: path={path}, key={filename}, objects={objects}")

    def get_image_details(self, filename):
        """Get comprehensive details for an image including EXIF, rating, tags, and label."""
        path = self.get_image_path(filename)
        if not path:
            return None

        details = {
            "exif": self.load_exif(path),
            "rating": self.get_rating(path),
            "tags": self.get_tags(path),
            "label": self.get_state(path),
            "objects": self.get_objects(path),
        }
        return details
