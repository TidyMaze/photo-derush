import sys
import logging
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QWidget, QTextEdit
from PySide6.QtGui import QIcon, QPixmap
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

def get_image_files(directory):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return [f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in image_exts]

def main():
    app = QApplication(sys.argv)
    last_dir = load_last_dir()
    dir_path = QFileDialog.getExistingDirectory(None, "Select Directory", last_dir or "")
    if not dir_path:
        return
    save_last_dir(dir_path)
    image_files = get_image_files(dir_path)
    image_paths = [os.path.join(dir_path, f) for f in image_files]

    win = QMainWindow()
    win.setWindowTitle("Photo App - Image Browser")
    win.resize(1000, 700)

    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    list_widget = QListWidget()
    list_widget.setIconSize(QPixmap(128, 128).size())
    exif_view = QTextEdit()
    exif_view.setReadOnly(True)
    exif_view.setMinimumHeight(120)
    exif_view.setPlaceholderText("Select an image to view EXIF data.")

    def show_exif_for_item(item):
        if not item:
            exif_view.setText("")
            return
        idx = list_widget.row(item)
        if idx < 0 or idx >= len(image_paths):
            exif_view.setText("")
            return
        path = image_paths[idx]
        try:
            img = Image.open(path)
            getexif = getattr(img, "_getexif", None)
            if not callable(getexif):
                exif_view.setText("No EXIF data found.")
                return
            exif_data = getexif()
            if not exif_data:
                exif_view.setText("No EXIF data found.")
                return
            exif = {}
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, str(tag))
                exif[tag_name] = value
            lines = [f"{k}: {v}" for k, v in sorted(exif.items())]
            exif_view.setText("\n".join(lines))
        except Exception as e:
            logging.warning(f"Could not read EXIF for {path}: {e}")
            exif_view.setText("No EXIF data or not a photo.")

    if not image_paths:
        layout.addWidget(QLabel("No images found in the selected directory."))
    else:
        for path in image_paths:
            item = QListWidgetItem(os.path.basename(path))
            try:
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    icon = QIcon(pixmap.scaled(128, 128))
                    item.setIcon(icon)
            except Exception as e:
                logging.warning(f"Could not load image {path}: {e}")
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        layout.addWidget(exif_view)
        list_widget.currentItemChanged.connect(show_exif_for_item)

    win.setCentralWidget(central_widget)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
