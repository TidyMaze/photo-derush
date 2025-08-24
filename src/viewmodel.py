from PySide6.QtCore import QObject, Signal, Slot, QThread
from src.model import ImageModel
from src.cache import ThumbnailCache
import subprocess
import sys
import logging
import os

class ImageLoaderWorker(QObject):
    image_found = Signal(str)
    thumbnail_loaded = Signal(str, object)
    progress = Signal(int, int)
    finished = Signal()

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        files = self.model.get_image_files()
        total = len(files)
        for idx, filename in enumerate(files):
            if self._abort:
                break
            self.image_found.emit(filename)
            thumb = self.model.load_thumbnail(self.model.get_image_path(filename))
            logging.info(f"ImageLoaderWorker: loaded thumbnail for {filename}: {'yes' if thumb else 'no'}")
            self.thumbnail_loaded.emit(filename, thumb)  # Emit filename, not path
            self.progress.emit(idx + 1, total)
        self.finished.emit()

class PhotoViewModel(QObject):
    """ViewModel: Exposes all state and actions for the View. No UI code."""
    images_changed = Signal(list)
    image_added = Signal(str, int)  # filename, index
    exif_changed = Signal(dict)
    progress_changed = Signal(int, int)
    thumbnail_loaded = Signal(str, object)  # path, QImage or PIL Image
    selected_image_changed = Signal(str)
    has_selected_image_changed = Signal(bool)
    rating_changed = Signal(int)
    tags_changed = Signal(list)

    def __init__(self, directory, max_images=100):
        super().__init__()
        cache = ThumbnailCache()
        self.model = ImageModel(directory, max_images, cache=cache)
        self.images = []
        self.selected_image = None
        self.exif = {}
        self.max_images = max_images
        self._loader_thread = None
        self._loader_worker = None
        self.current_exts = self.model.allowed_exts.copy()
        self._has_selected_image = False
        self._rating = 0
        self._tags = []
        # Quick filter state
        self._quick_filter_rating = 0
        self._quick_filter_tag = ''
        self._quick_filter_date = ''

    @property
    def has_selected_image(self) -> bool:
        return self._has_selected_image

    @property
    def rating(self):
        return self._rating

    @property
    def tags(self):
        return self._tags

    def load_images(self):
        import logging
        logging.info(f"PhotoViewModel.load_images called. Directory: {self.model.directory}")
        self.images = []
        self.images_changed.emit(self.images)  # Clear grid at start
        self._loader_worker = ImageLoaderWorker(self.model)
        self._loader_thread = QThread()
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.image_found.connect(self._on_image_found)
        self._loader_worker.thumbnail_loaded.connect(self.thumbnail_loaded)
        self._loader_worker.progress.connect(self.progress_changed)
        self._loader_worker.finished.connect(self._on_loader_finished)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()

    def _on_loader_finished(self):
        self.images_changed.emit(self.images.copy())

    @Slot(str)
    def _on_image_found(self, filename):
        self.images.append(filename)
        self.image_added.emit(filename, len(self.images) - 1)

    def _update_rating_tags(self):
        if self.selected_image:
            self._rating = self.model.get_rating(self.selected_image)
            self._tags = self.model.get_tags(self.selected_image)
        else:
            self._rating = 0
            self._tags = []
        self.rating_changed.emit(self._rating)
        self.tags_changed.emit(self._tags)

    @Slot(str)
    def select_image(self, filename):
        path = self.model.get_image_path(filename)
        self.selected_image = path
        self.selected_image_changed.emit(path)
        self._has_selected_image = bool(path)
        self.has_selected_image_changed.emit(self._has_selected_image)
        self._update_rating_tags()
        exif = self.model.load_exif(path)
        self.exif = exif
        self.exif_changed.emit(exif)

    @Slot(int)
    def set_rating(self, rating):
        if self.selected_image:
            self.model.set_rating(self.selected_image, rating)
            self._rating = rating
            self.rating_changed.emit(rating)

    @Slot(list)
    def set_tags(self, tags):
        if self.selected_image:
            self.model.set_tags(self.selected_image, tags)
            self._tags = tags
            self.tags_changed.emit(tags)

    @Slot(str)
    def load_thumbnail(self, filename):
        # No-op: thumbnails are loaded by the worker now
        pass

    def set_file_types(self, exts):
        self.current_exts = exts
        self.model.set_allowed_exts(exts)
        self.load_images()

    def open_selected_in_viewer(self):
        if not self.selected_image:
            logging.warning("No image selected to open in viewer.")
            return
        path = self.selected_image
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            logging.error(f"Failed to open image in viewer: {e}")

    def set_quick_filter(self, rating=0, tag='', date=''):
        import logging
        logging.info(f"ViewModel.set_quick_filter: rating={rating}, tag='{tag}', date='{date}'")
        self._quick_filter_rating = rating
        self._quick_filter_tag = tag
        self._quick_filter_date = date
        self.apply_quick_filter()

    def apply_quick_filter(self):
        # Temporarily disable all filters: always show all images
        self.images = self.model.get_image_files()
        self.images_changed.emit(self.images)
        # Remove image_added emission to avoid double addition
        # for idx, filename in enumerate(self.images):
        #     self.image_added.emit(filename, idx)
