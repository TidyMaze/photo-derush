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
