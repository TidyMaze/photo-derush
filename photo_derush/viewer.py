from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication, QPushButton, QHBoxLayout, QWidget, QSizePolicy
from PySide6.QtCore import Qt
from .utils import pil2pixmap
from .image_manager import image_manager

def open_full_image_qt(img_path, on_keep=None, on_trash=None, on_unsure=None):
    dlg = QDialog()
    dlg.setWindowTitle("Full Image Viewer")
    dlg.setWindowFlag(Qt.WindowType.Window)
    dlg.setWindowState(Qt.WindowState.WindowFullScreen)
    img = image_manager.get_image(img_path)
    if img is None:
        # Show a placeholder dialog indicating failure
        placeholder = QLabel(f"Failed to load image:\n{img_path}")
        placeholder.setStyleSheet("color:#f55; background:#222; padding:40px; font-size:18px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(placeholder)
        dlg.setLayout(layout)
        dlg.exec()
        return
    screen = QApplication.primaryScreen().geometry()
    img = img.copy()
    img.thumbnail((screen.width(), screen.height()))
    pix = pil2pixmap(img)
    lbl = QLabel()
    lbl.setPixmap(pix)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    lbl.setScaledContents(True)
    # Buttons row
    buttons_widget = QWidget()
    btn_layout = QHBoxLayout(buttons_widget)
    btn_layout.setContentsMargins(20, 10, 20, 10)
    btn_layout.setSpacing(20)
    style_btn = "QPushButton { padding:12px 24px; font-size:16px; background:#444; color:#eee; border-radius:6px;} QPushButton:hover { background:#666; }"
    keep_btn = QPushButton("Keep")
    trash_btn = QPushButton("Trash")
    unsure_btn = QPushButton("Unsure")
    for b in (keep_btn, trash_btn, unsure_btn):
        b.setStyleSheet(style_btn)
    btn_layout.addStretch(1)
    btn_layout.addWidget(keep_btn)
    btn_layout.addWidget(trash_btn)
    btn_layout.addWidget(unsure_btn)
    btn_layout.addStretch(1)
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(lbl, 1)
    layout.addWidget(buttons_widget, 0)
    dlg.setLayout(layout)
    def close_event():
        dlg.accept()
    def wrap(cb):
        if cb:
            cb()
        close_event()
    keep_btn.clicked.connect(lambda: wrap(on_keep))
    trash_btn.clicked.connect(lambda: wrap(on_trash))
    unsure_btn.clicked.connect(lambda: wrap(on_unsure))
    # Shortcuts via keyPress
    def key_handler(e):
        key = e.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            close_event()
        elif key in (Qt.Key.Key_K, Qt.Key.Key_1):
            wrap(on_keep)
        elif key in (Qt.Key.Key_T, Qt.Key.Key_0):
            wrap(on_trash)
        elif key in (Qt.Key.Key_U, Qt.Key.Key_2):
            wrap(on_unsure)
    dlg.keyPressEvent = key_handler
    # Disable click-to-close on main image (could still close with ESC/Q)
    dlg.exec()
