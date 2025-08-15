"""
PySide6 port of the Lightroom UI from main.py (Tkinter version).
- Main window: QMainWindow with left (image grid) and right (info) panels
- Image grid: QScrollArea + QGridLayout, thumbnails, selection, metrics overlay
- Full image viewer: QDialog or QMainWindow, closes on click/ESC
- All event handling and image display is Qt idiomatic
"""
import os
import hashlib
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QLabel, QGridLayout, QSplitter, QDialog, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PIL import Image
import cv2

MAX_IMAGES = 200

# --- Utility: PIL Image to QPixmap ---
def pil2pixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

# --- Full Image Viewer ---
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

# --- Main Lightroom UI ---
def show_lightroom_ui_qt(image_paths, directory, trashed_paths=None, trashed_dir=None):
    app = QApplication.instance() or QApplication([])
    # Load and apply QDarkStyle stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "qdarkstyle.qss")
    with open(qss_path, "r") as f:
        app.setStyleSheet(f.read())
    win = QMainWindow()
    win.setWindowTitle("Photo Derush (Qt)")
    win.resize(1400, 800)
    status = QStatusBar()
    win.setStatusBar(status)
    # Splitter for left/right panels
    splitter = QSplitter()
    win.setCentralWidget(splitter)
    # Left: image grid
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    left_layout.addWidget(scroll)
    grid_container = QWidget()
    grid_container.setStyleSheet("background-color: #222;")
    grid = QGridLayout(grid_container)
    scroll.setWidget(grid_container)
    # Right: info panel
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    info_label = QLabel(("This is the right panel.\n" * 50).strip())
    info_label.setStyleSheet("color: #aaa; background: #222; font-size: 14pt;")
    info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    right_layout.addWidget(info_label)
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setSizes([1000, 400])
    # --- Image grid population ---
    THUMB_SIZE = 160
    num_images = min(MAX_IMAGES, len(image_paths))
    image_labels = []
    top_labels = []
    bottom_labels = []
    blur_labels = []
    metrics_cache = {}
    # --- Group images by hash ---
    hash_to_images = {}
    image_hashes = {}
    for img_name in image_paths[:num_images]:
        img_path = os.path.join(directory, img_name)
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                sha256_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        else:
            sha256_hash = "MISSING"
        image_hashes[img_name] = sha256_hash
        hash_to_images.setdefault(sha256_hash, []).append(img_name)
    # --- Populate grid by group ---
    row = 0
    col_count = 5
    for sha256_hash, group in hash_to_images.items():
        # Add group header
        group_label = QLabel(f"Hash group: {sha256_hash}")
        group_label.setStyleSheet("color: #3daee9; background: #232629; font-weight: bold; font-size: 12pt;")
        grid.addWidget(group_label, row, 0, 1, col_count)
        row += 1
        for i, img_name in enumerate(group):
            img_path = os.path.join(directory, img_name)
            thumb_path = os.path.join(directory, 'thumbnails', img_name)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
            else:
                img = Image.open(img_path)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(thumb_path)
            pix = pil2pixmap(img)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            lbl.setStyleSheet("background: #444; border: 2px solid #444;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Top label
            blur_score = compute_blur_score(img_path)
            if blur_score is not None:
                top_label = QLabel(f"Blur: {blur_score:.1f}")
            else:
                top_label = QLabel("")
            top_label.setStyleSheet("color: red; background: #222;")
            # Bottom label
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nSHA256: {sha256_hash}")
            bottom_label.setStyleSheet("color: white; background: #222;")
            # Blur/metrics label
            blur_label = QLabel("")
            blur_label.setStyleSheet("color: yellow; background: #222;")
            # Mouse events
            idx = len(image_labels)
            def enterEventFactory(idx=idx):
                return lambda e: show_metrics(idx)
            def leaveEventFactory(idx=idx):
                return lambda e: hide_metrics(idx)
            lbl.enterEvent = enterEventFactory()
            lbl.leaveEvent = leaveEventFactory()
            # Selection and double-click
            def mousePressEventFactory(idx=idx, label=lbl):
                def handler(e: QMouseEvent):
                    for l in image_labels:
                        l.setStyleSheet("background: #444; border: 2px solid #444;")
                    label.setStyleSheet("background: red; border: 2px solid red;")
                return handler
            def mouseDoubleClickEventFactory(img_path=img_path):
                return lambda e: open_full_image_qt(img_path)
            lbl.mousePressEvent = mousePressEventFactory(idx, lbl)
            lbl.mouseDoubleClickEvent = mouseDoubleClickEventFactory(img_path)
            # Add to grid
            col = i % col_count
            grid.addWidget(lbl, row*4, col)
            grid.addWidget(top_label, row*4+1, col)
            grid.addWidget(bottom_label, row*4+2, col)
            grid.addWidget(blur_label, row*4+3, col)
            image_labels.append(lbl)
            top_labels.append(top_label)
            bottom_labels.append(bottom_label)
            blur_labels.append(blur_label)
            if col == col_count - 1:
                row += 1
        # Add spacing after each group
        row += 2
    status.showMessage(f"Loaded {num_images} images")
    win.show()
    app.exec()
    win.closeEvent = lambda event: app.quit()

# --- Metrics functions (copied from main.py) ---
def compute_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_sharpness_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = {}
    features['variance_laplacian'] = cv2.Laplacian(img, cv2.CV_64F).var()
    features['tenengrad'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Tenengrad
    features['brenner'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Brenner
    features['wavelet_energy'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Wavelet energy
    return features
