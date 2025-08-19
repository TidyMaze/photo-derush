from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication, QPushButton, QHBoxLayout, QWidget, QSizePolicy, QScrollArea
from PySide6.QtCore import Qt, QSize
from .utils import pil2pixmap
from .image_manager import image_manager
import logging

logger = logging.getLogger(__name__)

WINDOW_SCREEN_RATIO = 0.6  # window occupies 60% of available screen in each dimension
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600
# Add absolute maximum dimensions to prevent oversized windows
MAX_WINDOW_WIDTH = 1600
MAX_WINDOW_HEIGHT = 1200

class EmbeddedImageViewer(QWidget):
    """Image viewer panel embedded inside the main window (no separate dialog)."""
    def __init__(self, parent_main_window, image_sequence, start_index, on_keep, on_trash, on_unsure, on_index_change):
        super().__init__()
        self.parent_main_window = parent_main_window
        self.image_sequence = image_sequence
        self.current_index = max(0, min(start_index, len(image_sequence)-1))
        self.on_keep = on_keep
        self.on_trash = on_trash
        self.on_unsure = on_unsure
        self.on_index_change = on_index_change
        self.orig_pix = None
        self._filmstrip_count = 9  # odd number, center current
        self._filmstrip_labels = []
        self._build_ui()
        self.load_current()

    def sizeHint(self):  # allow a reasonable size hint
        return QSize(1200, 800)

    def _build_ui(self):
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Buttons
        self.buttons_widget = QWidget()
        btn_layout = QHBoxLayout(self.buttons_widget)
        btn_layout.setContentsMargins(20, 10, 20, 10)
        btn_layout.setSpacing(20)
        style_btn = "QPushButton { padding:10px 18px; font-size:14px; background:#444; color:#eee; border-radius:5px;} QPushButton:hover { background:#666; }"
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.keep_btn = QPushButton("Keep (K/1)")
        self.trash_btn = QPushButton("Trash (T/0)")
        self.unsure_btn = QPushButton("Unsure (U/2)")
        self.exit_btn = QPushButton("Back (Esc/Q)")
        for b in (self.prev_btn, self.next_btn, self.keep_btn, self.trash_btn, self.unsure_btn, self.exit_btn):
            b.setStyleSheet(style_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addSpacing(30)
        btn_layout.addWidget(self.keep_btn)
        btn_layout.addWidget(self.trash_btn)
        btn_layout.addWidget(self.unsure_btn)
        btn_layout.addSpacing(30)
        btn_layout.addWidget(self.exit_btn)
        btn_layout.addStretch(1)
        # Filmstrip (scrollable row of thumbnails)
        self.filmstrip_container = QWidget()
        fs_layout = QHBoxLayout(self.filmstrip_container)
        fs_layout.setContentsMargins(8,4,8,4)
        fs_layout.setSpacing(6)
        for _ in range(self._filmstrip_count):
            thumb = QLabel()
            thumb.setFixedSize(96, 72)
            thumb.setStyleSheet("background:#111; border:2px solid #333;")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._filmstrip_labels.append(thumb)
            fs_layout.addWidget(thumb)
        self.filmstrip_scroll = QScrollArea()
        self.filmstrip_scroll.setWidgetResizable(True)
        self.filmstrip_scroll.setFixedHeight(90)
        inner = QWidget()
        inner_layout = QHBoxLayout(inner)
        inner_layout.setContentsMargins(0,0,0,0)
        inner_layout.addWidget(self.filmstrip_container)
        self.filmstrip_scroll.setWidget(inner)

        # Insert filmstrip between image and buttons
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.lbl, 1)
        layout.addWidget(self.filmstrip_scroll, 0)
        layout.addWidget(self.buttons_widget, 0)

        # Connections
        self.prev_btn.clicked.connect(lambda: self.navigate(-1))
        self.next_btn.clicked.connect(lambda: self.navigate(1))
        self.keep_btn.clicked.connect(self._wrap(self.on_keep, advance=True))
        self.trash_btn.clicked.connect(self._wrap(self.on_trash, advance=True))
        self.unsure_btn.clicked.connect(self._wrap(self.on_unsure, advance=True))
        self.exit_btn.clicked.connect(self._exit)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _wrap(self, cb, advance=False):
        def inner():
            progressed = False
            if cb:
                try:
                    cb()
                    progressed = True
                except Exception as e:  # noqa: PERF203
                    logger.warning('[Viewer] action callback failed: %s', e)
            if advance and progressed:
                # Advance only if not at last image
                if self.current_index < len(self.image_sequence) - 1:
                    self.navigate(1)
        return inner

    def _exit(self):
        # Restore main window splitter
        if hasattr(self.parent_main_window, '_restore_from_viewer'):
            self.parent_main_window._restore_from_viewer()

    def keyPressEvent(self, e):  # noqa: N802
        key = e.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self._exit()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_H):
            self.navigate(-1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_L, Qt.Key.Key_Space):
            self.navigate(1)
        elif key in (Qt.Key.Key_K, Qt.Key.Key_1):
            self._wrap(self.on_keep, advance=True)()
        elif key in (Qt.Key.Key_T, Qt.Key.Key_0):
            self._wrap(self.on_trash, advance=True)()
        elif key in (Qt.Key.Key_U, Qt.Key.Key_2):
            self._wrap(self.on_unsure, advance=True)()
        else:
            super().keyPressEvent(e)

    def navigate(self, delta):
        new_index = self.current_index + delta
        if 0 <= new_index < len(self.image_sequence):
            self.current_index = new_index
            self.load_current()

    def _update_filmstrip(self):
        if not self.image_sequence:
            return
        half = self._filmstrip_count // 2
        start = max(0, self.current_index - half)
        end = min(len(self.image_sequence), start + self._filmstrip_count)
        # Adjust start if near end
        if end - start < self._filmstrip_count:
            start = max(0, end - self._filmstrip_count)
        seq_slice = list(range(start, end))
        for lbl, idx in zip(self._filmstrip_labels, seq_slice):
            path = self.image_sequence[idx]
            pil_img = image_manager.get_thumbnail(path, (160, 120))
            if pil_img is None:
                lbl.setText('X')
                continue
            pm = pil2pixmap(pil_img).scaled(lbl.width(), lbl.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(pm)
            border = '#4caf50' if idx == self.current_index else '#333'
            lbl.setStyleSheet(f"background:#111; border:3px solid {border};")
            def _make_handler(target=idx):
                def _h(_e):
                    self.current_index = target
                    self.load_current()
                return _h
            lbl.mousePressEvent = _make_handler()
        # Clear remaining labels if any
        for lbl in self._filmstrip_labels[len(seq_slice):]:
            lbl.clear(); lbl.setStyleSheet("background:#111; border:2px solid #222;")

    def load_current(self):
        path = self.image_sequence[self.current_index]
        # Determine target size for thumbnail (approx current widget)
        avail_w = max(100, self.width() or 1200)
        btn_h = self.buttons_widget.sizeHint().height() or 80
        avail_h = max(100, (self.height() or 800) - btn_h)
        pil_img = image_manager.get_thumbnail(path, (avail_w, avail_h))
        if pil_img is None:
            self.lbl.setText(f"Failed to load image:\n{path}")
            return
        self.orig_pix = pil2pixmap(pil_img)
        self._rescale()
        # Nav buttons
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_sequence)-1)
        if self.on_index_change:
            try:
                self.on_index_change(path, self.current_index)
            except Exception:
                pass
        self._update_filmstrip()

    def resizeEvent(self, ev):  # noqa: N802
        self._rescale()
        return super().resizeEvent(ev)

    def _rescale(self):
        if self.orig_pix is None:
            return
        btn_h = self.buttons_widget.height() or self.buttons_widget.sizeHint().height() or 0
        filmstrip_h = self.filmstrip_scroll.height() or self.filmstrip_scroll.sizeHint().height() or 0
        avail_w = max(10, self.width())
        avail_h = max(10, self.height() - btn_h - filmstrip_h)
        scaled = self.orig_pix.scaled(avail_w, avail_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(scaled)

# Legacy dialog approach retained for backward compatibility (tests may rely on function call)
def open_full_image_qt(img_path, on_keep=None, on_trash=None, on_unsure=None, image_sequence=None, start_index=None, on_index_change=None, parent_main_window=None):
    # If parent_main_window provided -> embed instead of dialog
    if parent_main_window is not None:
        seq = image_sequence or [img_path]
        if start_index is None:
            try:
                start_index = seq.index(img_path)
            except ValueError:
                start_index = 0
        viewer = EmbeddedImageViewer(parent_main_window, seq, start_index, on_keep, on_trash, on_unsure, on_index_change)
        # Ask main window to display
        if hasattr(parent_main_window, '_show_embedded_viewer'):
            parent_main_window._show_embedded_viewer(viewer)
        return viewer
    # Fallback dialog mode (legacy)
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
    # Not using true full screen to avoid covering entire display / multi-monitor issues
    screen_geo = QApplication.primaryScreen().availableGeometry()
    target_w = max(MIN_WINDOW_WIDTH, int(screen_geo.width() * WINDOW_SCREEN_RATIO))
    target_h = max(MIN_WINDOW_HEIGHT, int(screen_geo.height() * WINDOW_SCREEN_RATIO))
    # Clamp to maximum dimensions
    target_w = min(target_w, MAX_WINDOW_WIDTH)
    target_h = min(target_h, MAX_WINDOW_HEIGHT)
    # Center the window
    x = screen_geo.x() + (screen_geo.width() - target_w) // 2
    y = screen_geo.y() + (screen_geo.height() - target_h) // 2
    dlg.setGeometry(x, y, target_w, target_h)
    dlg.setMinimumSize(int(MIN_WINDOW_WIDTH*0.8), int(MIN_WINDOW_HEIGHT*0.8))

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

    def _scale_current_pixmap():
        nonlocal orig_pix
        if orig_pix is None:
            return
        btn_h = buttons_widget.height() or buttons_widget.sizeHint().height() or 0
        avail_w = max(10, dlg.width())
        avail_h = max(10, dlg.height() - btn_h)
        try:
            scaled = orig_pix.scaled(avail_w, avail_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(scaled)
        except Exception as e:  # noqa: PERF203
            logger.debug('[Viewer] Scaling failed: %s', e)

    def load_current():
        nonlocal orig_pix, current_index
        path = image_sequence[current_index]
        # Determine target thumbnail size based on dialog dimensions
        est_btn_h = buttons_widget.sizeHint().height() or 80
        target_w = max(100, dlg.width())
        target_h = max(100, dlg.height() - est_btn_h)
        pil_img = image_manager.get_thumbnail(path, (target_w, target_h))
        if pil_img is None:
            lbl.setText(f"Failed to load image:\n{path}")
            return
        orig_pix = pil2pixmap(pil_img)
        _scale_current_pixmap()
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

    def wrap(cb, advance=False):
        if not cb:
            return
        try:
            cb()
            if advance and current_index < len(image_sequence)-1:
                navigate(1)
        except Exception as e:  # noqa: PERF203
            logger.warning("[Viewer] action callback failed: %s", e)

    prev_btn.clicked.connect(lambda: navigate(-1))
    next_btn.clicked.connect(lambda: navigate(1))
    keep_btn.clicked.connect(lambda: wrap(on_keep, True))
    trash_btn.clicked.connect(lambda: wrap(on_trash, True))
    unsure_btn.clicked.connect(lambda: wrap(on_unsure, True))

    def key_handler(e):
        key = e.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            close_event()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_H):
            navigate(-1)
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_L, Qt.Key.Key_Space):
            navigate(1)
        elif key in (Qt.Key.Key_K, Qt.Key.Key_1):
            wrap(on_keep, True)
        elif key in (Qt.Key.Key_T, Qt.Key.Key_0):
            wrap(on_trash, True)
        elif key in (Qt.Key.Key_U, Qt.Key.Key_2):
            wrap(on_unsure, True)
    dlg.keyPressEvent = key_handler

    def resize_event(ev):
        _scale_current_pixmap()
        return QDialog.resizeEvent(dlg, ev)
    dlg.resizeEvent = resize_event

    load_current()
    dlg.exec()
