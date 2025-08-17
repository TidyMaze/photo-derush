from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication, QPushButton, QHBoxLayout, QWidget, QSizePolicy
from PySide6.QtCore import Qt
from .utils import pil2pixmap
from .image_manager import image_manager
import logging

logger = logging.getLogger(__name__)

def open_full_image_qt(img_path, on_keep=None, on_trash=None, on_unsure=None, image_sequence=None, start_index=None, on_index_change=None):
    """Fullscreen viewer with optional navigation.
    Parameters:
      img_path: initial image path (kept for backward compatibility)
      image_sequence: optional list of image (absolute) paths for navigation
      start_index: index in image_sequence corresponding to img_path
      on_index_change(new_path, new_index): callback when navigation occurs
    """
    # Normalize sequence
    if image_sequence is None:
        image_sequence = [img_path]
        start_index = 0
    else:
        if start_index is None:
            try:
                start_index = image_sequence.index(img_path)
            except ValueError:
                start_index = 0
    current_index = max(0, min(start_index, len(image_sequence)-1))

    dlg = QDialog()
    dlg.setWindowTitle("Full Image Viewer")
    dlg.setWindowFlag(Qt.WindowType.Window)
    dlg.setWindowState(Qt.WindowState.WindowFullScreen)

    # Widgets
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # Buttons row
    buttons_widget = QWidget()
    btn_layout = QHBoxLayout(buttons_widget)
    btn_layout.setContentsMargins(20, 10, 20, 10)
    btn_layout.setSpacing(20)
    style_btn = "QPushButton { padding:12px 20px; font-size:15px; background:#444; color:#eee; border-radius:6px;} QPushButton:hover { background:#666; }"
    prev_btn = QPushButton("◀ Prev")
    next_btn = QPushButton("Next ▶")
    keep_btn = QPushButton("Keep")
    trash_btn = QPushButton("Trash")
    unsure_btn = QPushButton("Unsure")
    for b in (prev_btn, next_btn, keep_btn, trash_btn, unsure_btn):
        b.setStyleSheet(style_btn)
    btn_layout.addStretch(1)
    btn_layout.addWidget(prev_btn)
    btn_layout.addWidget(next_btn)
    btn_layout.addSpacing(30)
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

    # State
    orig_pix = None

    def load_current():
        nonlocal orig_pix, current_index
        path = image_sequence[current_index]
        pil_img = image_manager.get_image(path)
        if pil_img is None:
            lbl.setText(f"Failed to load image:\n{path}")
            return
        screen = QApplication.primaryScreen().geometry()
        pil_c = pil_img.copy()
        pil_c.thumbnail((screen.width(), screen.height()))
        orig_pix = pil2pixmap(pil_c)
        # Scale to fit preserving ratio
        scaled = orig_pix.scaled(lbl.width() or screen.width(), lbl.height() or screen.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        lbl.setPixmap(scaled)
        # Enable / disable nav buttons
        prev_btn.setEnabled(current_index > 0)
        next_btn.setEnabled(current_index < len(image_sequence)-1)
        if on_index_change:
            try:
                on_index_change(path, current_index)
            except Exception as e:  # noqa: PERF203
                logger.debug("[Viewer] on_index_change failed: %s", e)

    def navigate(delta):
        nonlocal current_index
        new_index = current_index + delta
        if 0 <= new_index < len(image_sequence):
            current_index = new_index
            load_current()

    def close_event():
        dlg.accept()

    def wrap(cb):
        if cb:
            try:
                # Provide current path for convenience if callback wants introspection
                cb()
            except Exception as e:  # noqa: PERF203
                logger.warning("[Viewer] action callback failed: %s", e)

    prev_btn.clicked.connect(lambda: navigate(-1))
    next_btn.clicked.connect(lambda: navigate(1))
    keep_btn.clicked.connect(lambda: wrap(on_keep))
    trash_btn.clicked.connect(lambda: wrap(on_trash))
    unsure_btn.clicked.connect(lambda: wrap(on_unsure))

    def key_handler(e):
        key = e.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            close_event()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_H):
            navigate(-1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_L, Qt.Key.Key_Space):
            navigate(1)
        elif key in (Qt.Key.Key_K, Qt.Key.Key_1):
            wrap(on_keep)
        elif key in (Qt.Key.Key_T, Qt.Key.Key_0):
            wrap(on_trash)
        elif key in (Qt.Key.Key_U, Qt.Key.Key_2):
            wrap(on_unsure)
    dlg.keyPressEvent = key_handler

    def resize_event(ev):
        try:
            if orig_pix is not None:
                scaled = orig_pix.scaled(lbl.width(), lbl.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                lbl.setPixmap(scaled)
        except Exception as e:  # noqa: PERF203
            logger.debug('[Viewer] Resize scaling failed: %s', e)
        return QDialog.resizeEvent(dlg, ev)
    dlg.resizeEvent = resize_event

    load_current()
    dlg.exec()
