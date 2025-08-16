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
from precompute import prepare_images_and_groups, MAX_IMAGES as PREP_MAX, list_images
import logging

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
        try:
            images = list_images(directory)
            subset = images[:max_images]
            logging.info("[AsyncLoad] (stream) Found %d images, streaming first %d", len(images), len(subset))
            # Set sorted images early
            def set_sorted():
                win.sorted_images = subset
            QTimer.singleShot(0, set_sorted)
            # Stream thumbnails quickly
            for img in subset:
                def add(img_name=img):
                    if hasattr(win, 'image_grid') and win.image_grid:
                        win.image_grid.add_image(img_name)
                QTimer.singleShot(0, add)
            # After streaming, do full hashing/grouping
            images2, image_info, stats = prepare_images_and_groups(directory, max_images)
            def apply_grouping():
                win.update_grouping(image_info)
                win.status.showMessage(f"Loaded {len(subset)} images (groups ready)")
            QTimer.singleShot(0, apply_grouping)
        except Exception as e:
            logging.exception("[AsyncLoad] Worker failed: %s", e)
            def fail():
                win.status.showMessage(f"Background load failed: {e}")
            QTimer.singleShot(0, fail)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.exec()
