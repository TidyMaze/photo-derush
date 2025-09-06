from PySide6.QtCore import QObject, Signal, Slot, QThread
from .model import ImageModel
from .cache import ThumbnailCache
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
            full_path = self.model.get_image_path(filename)
            thumb = self.model.load_thumbnail(full_path)
            logging.info(f"ImageLoaderWorker: loaded thumbnail for {filename}: {'yes' if thumb else 'no'}")
            self.thumbnail_loaded.emit(full_path, thumb)
            self.progress.emit(idx + 1, total)
        self.finished.emit()

class PhotoViewModel(QObject):
    """ViewModel: Exposes all state and actions for the View. No UI code."""
    images_changed = Signal(list)
    image_added = Signal(str, int)
    exif_changed = Signal(dict)
    progress_changed = Signal(int, int)
    thumbnail_loaded = Signal(str, object)
    selected_image_changed = Signal(str)
    has_selected_image_changed = Signal(bool)
    rating_changed = Signal(int)
    tags_changed = Signal(list)
    label_changed = Signal(str, str)
    state_changed = Signal(str)

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
        self._has_selected_image = False
        self._rating = 0
        self._tags = []
        self._label = None
        self._state = ''

    @property
    def has_selected_image(self) -> bool:
        return self._has_selected_image

    @property
    def state(self):
        return self._state

    @property
    def rating(self):
        return self._rating

    @property
    def tags(self):
        return self._tags

    @property
    def label(self):
        return self._label

    def load_images(self):
        logging.info(f"PhotoViewModel.load_images called. Directory: {self.model.directory}")
        self.images = []
        self.images_changed.emit(self.images)

        # Create and store strong references
        self._loader_worker = ImageLoaderWorker(self.model)
        self._loader_thread = QThread(self)

        # Move worker to thread
        self._loader_worker.moveToThread(self._loader_thread)

        # Connect signals properly
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.image_found.connect(self._on_image_found)
        self._loader_worker.thumbnail_loaded.connect(self.thumbnail_loaded)
        self._loader_worker.progress.connect(self.progress_changed)

        # Proper cleanup sequence
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._on_thread_finished)

        logging.info("Starting image loader thread")
        self._loader_thread.start()

    def set_label(self, label: str):
        if self.selected_image:
            filenames = [os.path.basename(p) for p in self.selected_image]
            for filename in filenames:
                self.model.set_state(filename, label)
                self.label_changed.emit(filename, label)
            self._label = label

    def _on_thread_finished(self):
        """Called when the loader thread has completely finished"""
        logging.info("Loader thread finished")
        if self._loader_thread:
            self._loader_thread.deleteLater()
            self._loader_thread = None
        self._loader_worker = None

    def cleanup(self):
        """Properly stop and wait for the loader thread"""
        if self._loader_thread and self._loader_thread.isRunning():
            logging.info("Stopping loader thread...")
            if self._loader_worker:
                self._loader_worker.abort()
            self._loader_thread.quit()
            if not self._loader_thread.wait(5000):
                logging.warning("Thread did not stop gracefully, forcing termination")
                self._loader_thread.terminate()
                self._loader_thread.wait(1000)
            logging.info("Loader thread stopped")
        else:
            logging.info("Loader thread not running, no cleanup needed")

        if self._loader_thread:
            assert not self._loader_thread.isRunning(), "Thread should not be running after cleanup"

    @Slot(str)
    def _on_image_found(self, filename):
        self.images.append(filename)
        self.image_added.emit(filename, len(self.images) - 1)

    def _update_rating_tags(self):
        if self.selected_image:
            self._rating = self.model.get_rating(self.selected_image)
            self._tags = self.model.get_tags(self.selected_image)
            self._state = self.model.get_state(self.selected_image)
        else:
            self._rating = 0
            self._tags = []
            self._state = ''
        self.rating_changed.emit(self._rating)
        self.tags_changed.emit(self._tags)
        self.state_changed.emit(self._state)

    @Slot(str)
    def select_image(self, filename: str):
        full_path = self.model.get_image_path(filename)
        self.selected_image = [full_path]

        details = self.model.get_image_details(filename)
        if details:
            self.exif = details.get('exif', {})
            self._rating = details.get('rating', 0)
            self._tags = details.get('tags', [])
            self._label = details.get('label', None)
        else:
            self.exif = {}
            self._rating = 0
            self._tags = []
            self._label = None

        self.exif_changed.emit(self.exif)
        self.rating_changed.emit(self.rating)
        self.tags_changed.emit(self.tags)
        if self._label:
            self.label_changed.emit(filename, self._label)

        if not self._has_selected_image:
            self._has_selected_image = True
            self.has_selected_image_changed.emit(True)

        self.selected_image_changed.emit(full_path)

    @Slot(int)
    @Slot(int)
    def set_rating(self, rating: int):
        if self.selected_image:
            self.model.set_rating(self.selected_image, rating)
            self._rating = rating
            self.rating_changed.emit(self.rating)

    @Slot(list)
    def set_tags(self, tags: list):
        if self.selected_image:
            filenames = [os.path.basename(p) for p in self.selected_image]
            for filename in filenames:
                self.model.set_tags(filename, tags)
            self._tags = tags
            self.tags_changed.emit(self.tags)

    def load_thumbnail(self, filename: str):
        """Load thumbnail for a specific image file."""
        full_path = self.model.get_image_path(filename)
        return self.model.load_thumbnail(full_path)

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
