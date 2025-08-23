from PySide6.QtCore import QObject, Signal, Slot, QThread
from model import ImageModel
from cache import ThumbnailCache

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
            self.thumbnail_loaded.emit(filename, thumb)  # Emit filename, not path
            self.progress.emit(idx + 1, total)
        self.finished.emit()

class PhotoViewModel(QObject):
    images_changed = Signal(list)
    image_added = Signal(str, int)  # filename, index
    exif_changed = Signal(dict)
    progress_changed = Signal(int, int)
    thumbnail_loaded = Signal(str, object)  # path, QImage or PIL Image

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

    def load_images(self):
        self.images = []
        self.images_changed.emit(self.images)
        self._loader_worker = ImageLoaderWorker(self.model)
        self._loader_thread = QThread()
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.image_found.connect(self._on_image_found)
        self._loader_worker.thumbnail_loaded.connect(self.thumbnail_loaded)
        self._loader_worker.progress.connect(self.progress_changed)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()

    @Slot(str)
    def _on_image_found(self, filename):
        self.images.append(filename)
        self.images_changed.emit(self.images.copy())
        self.image_added.emit(filename, len(self.images) - 1)

    @Slot(str)
    def select_image(self, filename):
        path = self.model.get_image_path(filename)
        self.selected_image = path
        exif = self.model.load_exif(path)
        self.exif = exif
        self.exif_changed.emit(exif)

    @Slot(str)
    def load_thumbnail(self, filename):
        # No-op: thumbnails are loaded by the worker now
        pass
