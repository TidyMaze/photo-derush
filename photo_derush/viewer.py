from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication
from PySide6.QtCore import Qt
from PIL import Image
from .utils import pil2pixmap

def open_full_image_qt(img_path):
    dlg = QDialog()
    dlg.setWindowTitle("Full Image Viewer")
    dlg.setWindowFlag(Qt.WindowType.Window)
    dlg.setWindowState(Qt.WindowState.WindowFullScreen)
    img = Image.open(img_path)
    screen = QApplication.primaryScreen().geometry()
    img.thumbnail((screen.width(), screen.height()))
    pix = pil2pixmap(img)
    lbl = QLabel()
    lbl.setPixmap(pix)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout = QVBoxLayout()
    layout.addWidget(lbl)
    dlg.setLayout(layout)
    def close_event(*_):
        dlg.accept()
    lbl.mousePressEvent = lambda e: close_event()
    dlg.keyPressEvent = lambda e: close_event() if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q) else None
    dlg.exec()

