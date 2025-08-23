import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PIL import Image, ExifTags
import os
import json

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')

def load_last_dir():
    try:
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
            last_dir = data.get('last_dir')
            if last_dir and os.path.isdir(last_dir):
                return last_dir
    except Exception as e:
        logging.info(f"No previous config or invalid: {e}")
    return None

def save_last_dir(path):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'last_dir': path}, f)
    except Exception as e:
        logging.warning(f"Could not save config: {e}")

class FileScanner(QThread):
    files_found = Signal(list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def run(self):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        files = [f for f in os.listdir(self.directory)
                 if os.path.splitext(f)[1].lower() in image_exts]
        self.files_found.emit(files)

class ImageLoader(QThread):
    image_loaded = Signal(QPixmap, dict, dict)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        info = {}
        exif = {}
        try:
            stat = os.stat(self.path)
            info['size'] = stat.st_size
            info['mtime'] = stat.st_mtime
        except Exception as e:
            logging.warning(f"Could not stat file: {e}")
            info['size'] = None
            info['mtime'] = None
        pixmap = QPixmap(self.path)
        if not pixmap.isNull():
            info['width'] = pixmap.width()
            info['height'] = pixmap.height()
            pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            info['width'] = None
            info['height'] = None
        # EXIF extraction
        try:
            img = Image.open(self.path)
            raw_exif = img._getexif()
            if raw_exif:
                for tag, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    exif[tag_name] = value
        except Exception as e:
            logging.info(f"No EXIF or failed to read EXIF: {e}")
        self.image_loaded.emit(pixmap, info, exif)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Photo App")
        self.resize(400, 650)
        self.list_widget = QListWidget()
        self.button = QPushButton("Select Directory")
        self.button.clicked.connect(self.select_directory)
        self.preview_label = QLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(270)
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.exif_label = QLabel("")
        self.exif_label.setAlignment(Qt.AlignLeft)
        self.exif_label.setWordWrap(True)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.exif_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.scanner = None
        self.current_dir = None
        self.image_loader = None
        self.last_dir = load_last_dir()
        if self.last_dir:
            self.load_directory(self.last_dir)
        self.list_widget.currentItemChanged.connect(self.preview_selected_image)
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", self.last_dir or "")
        if dir_path:
            self.load_directory(dir_path)
            save_last_dir(dir_path)
    def load_directory(self, dir_path):
        self.list_widget.clear()
        self.button.setEnabled(False)
        self.current_dir = dir_path
        self.scanner = FileScanner(dir_path)
        self.scanner.files_found.connect(self.display_files)
        self.scanner.finished.connect(lambda: self.button.setEnabled(True))
        self.scanner.start()
    def display_files(self, files):
        self.list_widget.addItems(files)
    def preview_selected_image(self, current, previous):
        if not current or not self.current_dir:
            self.preview_label.setText("Select an image to preview")
            self.preview_label.setPixmap(QPixmap())
            self.info_label.setText("")
            self.exif_label.setText("")
            return
        image_path = os.path.join(self.current_dir, current.text())
        self.preview_label.setText("Loading...")
        self.info_label.setText("")
        self.exif_label.setText("")
        self.image_loader = ImageLoader(image_path)
        self.image_loader.image_loaded.connect(self.show_image)
        self.image_loader.start()
    def show_image(self, pixmap, info, exif):
        if pixmap and not pixmap.isNull():
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setText("")
        else:
            self.preview_label.setText("Failed to load image")
        # File info
        size = info.get('size')
        mtime = info.get('mtime')
        width = info.get('width')
        height = info.get('height')
        size_str = f"{size/1024:.1f} KB" if size else "?"
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S') if mtime else "?"
        dim_str = f"{width} x {height}" if width and height else "?"
        self.info_label.setText(f"Size: {size_str}\nModified: {mtime_str}\nDimensions: {dim_str}")
        # EXIF info
        if exif:
            keys = [
                'DateTimeOriginal', 'Make', 'Model', 'LensModel', 'ExposureTime',
                'FNumber', 'ISOSpeedRatings', 'FocalLength', 'Software'
            ]
            exif_lines = [f"{k}: {exif[k]}" for k in keys if k in exif]
            if not exif_lines:
                exif_lines = ["No key EXIF fields found."]
            self.exif_label.setText("EXIF:\n" + "\n".join(exif_lines))
        else:
            self.exif_label.setText("No EXIF data.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
