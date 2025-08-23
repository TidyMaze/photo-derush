import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit, QProgressBar, QGridLayout, QScrollArea
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import QObject, Signal, QThread
from PIL import Image, ExifTags
import os
import json

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')

def load_last_dir():
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)
        last_dir = data.get('last_dir')
        if last_dir and os.path.isdir(last_dir):
            return last_dir
        raise FileNotFoundError

def save_last_dir(path):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'last_dir': path}, f)
    except Exception as e:
        logging.warning(f"Could not save config: {e}")

def get_image_files(directory):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return [f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in image_exts]

class ImageLoaderWorker(QObject):
    image_loaded = Signal(str, QIcon)
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def load_images(self):
        total = len(self.image_paths)
        qimage_format = getattr(QImage, "Format_RGBA8888", getattr(QImage, "Format_ARGB32", QImage.Format_RGB888))
        for idx, path in enumerate(self.image_paths, 1):
            logging.info(f"[BG] Loading image: {path}")
            img = Image.open(path)
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img_cropped = img.crop((left, top, right, bottom)).resize((128, 128), Image.Resampling.LANCZOS)
            img_cropped = img_cropped.convert("RGBA")
            data = img_cropped.tobytes("raw", "RGBA")
            qimg = QImage(data, 128, 128, qimage_format)
            pixmap = QPixmap.fromImage(qimg)
            icon = QIcon(pixmap)
            self.image_loaded.emit(path, icon)
            self.progress.emit(idx, total)
        self.finished.emit()

class ExifLoaderWorker(QObject):
    exif_loaded = Signal(str, dict)
    exif_error = Signal(str, str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._abort = False

    def abort(self):
        self._abort = True

    def load_exif(self):
        import time
        start = time.time()
        img = Image.open(self.path)
        getexif = getattr(img, "_getexif", None)
        if not callable(getexif):
            self.exif_loaded.emit(self.path, {})
            logging.info(f"No _getexif for {self.path} (took {time.time()-start:.3f}s)")
            return
        exif_data = getexif()
        exif_time = time.time() - start
        logging.info(f"[BG] _getexif for {self.path} took {exif_time:.3f}s")
        if not exif_data:
            self.exif_loaded.emit(self.path, {})
            return
        exif = {}
        if isinstance(exif_data, dict):
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, str(tag))
                exif[tag_name] = value
        if self._abort:
            logging.info(f"EXIF load for {self.path} aborted.")
            return
        self.exif_loaded.emit(self.path, exif)

class GuiUpdater(QObject):
    def __init__(self, exif_view, get_last_exif_path, get_exif_worker_thread, parent=None):
        super().__init__(parent)
        self.exif_view = exif_view
        self.get_last_exif_path = get_last_exif_path
        self.get_exif_worker_thread = get_exif_worker_thread

    def update_exif(self, loaded_path, exif):
        last_exif_path = self.get_last_exif_path()
        exif_worker_thread = self.get_exif_worker_thread()
        if last_exif_path != loaded_path:
            logging.info(f"Stale EXIF result for {loaded_path}, ignoring.")
            return
        if not exif:
            self.exif_view.setText("No EXIF data found.")
        else:
            lines = [f"{k}: {v}" for k, v in sorted(exif.items())]
            self.exif_view.setText("\n".join(lines))
        if exif_worker_thread is not None:
            exif_worker_thread.quit()
            exif_worker_thread.wait()

class UiUpdater(QObject):
    add_image = Signal(str, QIcon)
    update_progress = Signal(int, int)

def main():
    logging.info("[DEBUG] Entered main()")
    try:
        logging.info("Starting QApplication...")
        app = QApplication(sys.argv)
        last_dir = load_last_dir()
        logging.info(f"Loaded last_dir: {last_dir}")
        dir_path = last_dir
        if not dir_path:
            logging.info("No previous directory found. Exiting.")
            return
        save_last_dir(dir_path)
        logging.info(f"Saved last_dir: {dir_path}")
        image_files = get_image_files(dir_path)
        logging.info(f"Found {len(image_files)} image files.")
        image_paths = [os.path.join(dir_path, f) for f in image_files]

        win = QMainWindow()
        win.setWindowTitle("Photo App - Image Browser")
        win.resize(1000, 700)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        exif_view = QTextEdit()
        exif_view.setReadOnly(True)
        exif_view.setMinimumHeight(120)
        exif_view.setPlaceholderText("Select an image to view EXIF data.")

        # Progress bar for image loading
        progress_bar = None
        if image_paths:
            progress_bar = QProgressBar()
            progress_bar.setMinimum(0)
            progress_bar.setMaximum(len(image_paths))
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)
            progress_bar.setFormat("Loading images: %v/%m")
            layout.insertWidget(0, progress_bar)
            progress_bar.show()
            logging.info("Progress bar shown at the top of the layout.")

        # 2D grid for images
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_widget)
        thumb_size = 128
        cols = 5
        image_labels = []

        def add_image_to_grid(idx, path, icon):
            row = idx // cols
            col = idx % cols
            label = QLabel()
            label.setFixedSize(thumb_size, thumb_size)
            label.setScaledContents(True)
            if icon and not icon.isNull():
                label.setPixmap(icon.pixmap(thumb_size, thumb_size))
            label.setToolTip(os.path.basename(path))
            label.mousePressEvent = lambda e, p=path: show_exif_for_path(p)
            grid_layout.addWidget(label, row, col)
            image_labels.append(label)

        # State for EXIF worker/thread and last requested path
        exif_worker_thread = None
        exif_worker = None
        last_exif_path = None

        # Instantiate GuiUpdater in the main thread
        gui_updater = GuiUpdater(
            exif_view,
            lambda: last_exif_path,
            lambda: exif_worker_thread
        )

        def show_exif_for_path(path):
            nonlocal exif_worker_thread, exif_worker, last_exif_path
            if not path:
                exif_view.setText("")
                return
            last_exif_path = path
            exif_view.setText("Loading EXIF...")
            # Abort previous worker if running
            if exif_worker_thread is not None and hasattr(exif_worker_thread, 'isRunning') and exif_worker_thread.isRunning():
                if exif_worker is not None and hasattr(exif_worker, 'abort'):
                    exif_worker.abort()
                exif_worker_thread.quit()
                exif_worker_thread.wait()
            exif_worker = ExifLoaderWorker(path)
            exif_worker_thread = QThread()
            exif_worker.moveToThread(exif_worker_thread)
            exif_worker_thread.started.connect(exif_worker.load_exif)
            exif_worker.exif_loaded.connect(gui_updater.update_exif)
            exif_worker_thread.start()

        if not image_paths:
            logging.info("No images found in the selected directory.")
            layout.addWidget(QLabel("No images found in the selected directory."))
        else:
            layout.addWidget(scroll)
            layout.addWidget(exif_view)

            win.setCentralWidget(central_widget)
            logging.info("Showing main window...")
            win.show()
            logging.info("[DEBUG] Before QApplication exec")
            logging.info("Entering app event loop.")

            # Start image loading in background thread
            class GridImageAdder(QObject):
                def __init__(self, grid_adder_func):
                    super().__init__()
                    self.grid_adder_func = grid_adder_func
                    self.idx = 0
                def add(self, path, icon):
                    self.grid_adder_func(self.idx, path, icon)
                    self.idx += 1

            image_adder = GridImageAdder(add_image_to_grid)
            image_loader = ImageLoaderWorker(image_paths)
            image_thread = QThread()
            image_loader.moveToThread(image_thread)
            image_loader.image_loaded.connect(image_adder.add)
            image_thread.started.connect(image_loader.load_images)
            image_loader.finished.connect(image_thread.quit)
            image_loader.finished.connect(image_loader.deleteLater)
            image_thread.finished.connect(image_thread.deleteLater)
            if progress_bar:
                image_loader.progress.connect(lambda idx, total: progress_bar.setValue(idx))
                image_loader.finished.connect(progress_bar.hide)
            image_thread.start()

        logging.info("Entering app event loop.")
        sys.exit(app.exec())
    except Exception as e:
        logging.exception(f"[ERROR] Exception in main: {e}")

default_entry = __name__ == "__main__"
if default_entry:
    logging.info("[DEBUG] __main__ entry point reached")
    main()
