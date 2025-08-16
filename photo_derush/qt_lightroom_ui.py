"""
PySide6 port of the Lightroom UI from main.py (Tkinter version).
- Main window: QMainWindow with left (image grid) and right (info) panels
- Image grid: QScrollArea + QGridLayout, thumbnails, selection, metrics overlay
- Full image viewer: QDialog or QMainWindow, closes on click/ESC
- All event handling and image display is Qt idiomatic
"""
import sys
from PySide6.QtWidgets import QApplication
from .main_window import LightroomMainWindow
from .viewer import open_full_image_qt
from PySide6.QtCore import QTimer
import threading
from precompute import prepare_images_and_groups, MAX_IMAGES as PREP_MAX

def show_lightroom_ui_qt(image_paths, directory, trashed_paths=None, trashed_dir=None, on_window_opened=None, image_info=None):
    app = QApplication.instance() or QApplication(sys.argv)
    # Apply global darkstyle
    import os
    qss_path = os.path.join(os.path.dirname(__file__), 'qdarkstyle.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            app.setStyleSheet(f.read())
    def get_sorted_images():
        return image_paths
    win = LightroomMainWindow(image_paths, directory, get_sorted_images, image_info=image_info)
    win.show()
    app.exec()

def show_lightroom_ui_qt_async(directory, max_images=PREP_MAX):
    app = QApplication.instance() or QApplication(sys.argv)
    import os
    qss_path = os.path.join(os.path.dirname(__file__), 'qdarkstyle.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            app.setStyleSheet(f.read())
    def empty_sorted():
        return []
    win = LightroomMainWindow([], directory, empty_sorted, image_info={})
    win.status.showMessage("Preparing images in background…")
    win.show()
    def worker():
        images, image_info, stats = prepare_images_and_groups(directory, max_images)
        def apply():
            win.load_images(images[:max_images], image_info)
            win.status.showMessage(f"Loaded {len(images[:max_images])} images (async) - groups computed")
        QTimer.singleShot(0, apply)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.exec()
