from PySide6.QtCore import QObject, Signal, Slot
from model import ImageModel

class PhotoViewModel(QObject):
    images_changed = Signal(list)
    exif_changed = Signal(dict)
    progress_changed = Signal(int, int)
    thumbnail_loaded = Signal(str, object)  # path, QImage or PIL Image

    def __init__(self, directory, max_images=100):
        super().__init__()
        self.model = ImageModel(directory, max_images)
        self.images = []
        self.selected_image = None
        self.exif = {}
        self.max_images = max_images

    def load_images(self):
        self.images = self.model.get_image_files()
        self.images_changed.emit(self.images)

    @Slot(str)
    def select_image(self, filename):
        path = self.model.get_image_path(filename)
        self.selected_image = path
        exif = self.model.load_exif(path)
        self.exif = exif
        self.exif_changed.emit(exif)

    @Slot(str)
    def load_thumbnail(self, filename):
        path = self.model.get_image_path(filename)
        thumb = self.model.load_thumbnail(path)
        self.thumbnail_loaded.emit(path, thumb)

