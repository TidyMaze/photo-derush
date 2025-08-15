"""
PySide6 port of the Lightroom UI from main.py (Tkinter version).
- Main window: QMainWindow with left (image grid) and right (info) panels
- Image grid: QScrollArea + QGridLayout, thumbnails, selection, metrics overlay
- Full image viewer: QDialog or QMainWindow, closes on click/ESC
- All event handling and image display is Qt idiomatic
"""
import os
import hashlib
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QLabel, QGridLayout, QSplitter, QDialog, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PIL import Image
import cv2
import imagehash

logging.basicConfig(level=logging.INFO)

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

class LightroomMainWindow(QMainWindow):
    def closeEvent(self, event):
        QApplication.quit()
        super().closeEvent(event)

# --- Main Lightroom UI ---
def show_lightroom_ui_qt(image_paths, directory, trashed_paths=None, trashed_dir=None):
    logging.info("Starting Photo Derush Qt app...")
    logging.info(f"Image directory: {directory}")
    logging.info(f"Number of images: {len(image_paths)})")
    app = QApplication.instance() or QApplication([])
    # Load and apply QDarkStyle stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "qdarkstyle.qss")
    with open(qss_path, "r") as f:
        app.setStyleSheet(f.read())
    logging.info("QDarkStyle stylesheet applied.")
    win = LightroomMainWindow()
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
    def show_metrics(idx):
        img_path = os.path.join(directory, image_paths[idx])
        if img_path in metrics_cache:
            blur_score, sharpness_metrics, aesthetic_score = metrics_cache[img_path]
        else:
            blur_score = compute_blur_score(img_path)
            sharpness_metrics = compute_sharpness_features(img_path)
            aesthetic_score = 42
            metrics_cache[img_path] = (blur_score, sharpness_metrics, aesthetic_score)
        lines = [
            f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A",
            f"Laplacian: {sharpness_metrics['variance_laplacian']:.1f}",
            f"Tenengrad: {sharpness_metrics['tenengrad']:.1f}",
            f"Brenner: {sharpness_metrics['brenner']:.1f}",
            f"Wavelet: {sharpness_metrics['wavelet_energy']:.1f}",
            f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A"
        ]
        blur_labels[idx].setText("\n".join(lines))
    def hide_metrics(idx):
        blur_labels[idx].setText("")
    # --- Group images by similarity hash ---
    hash_threshold = 5  # Hamming distance threshold for similarity
    image_hashes = {}
    hash_to_filenames = {}
    for i, img_name in enumerate(image_paths[:num_images]):
        img_path = os.path.join(directory, img_name)
        if os.path.exists(img_path):
            phash = compute_perceptual_hash(img_path)
            if phash is not None:
                image_hashes[img_name] = phash
                hash_to_filenames.setdefault(str(phash), []).append(img_name)
            else:
                image_hashes[img_name] = None
        else:
            image_hashes[img_name] = None
        if i % 10 == 0 or i == num_images - 1:
            logging.info(f"Processed {i+1}/{num_images} images for perceptual hashing...")
    # --- Find similarity groups ---
    ungrouped = set(img_name for img_name, h in image_hashes.items() if h is not None)
    similarity_groups = []
    while ungrouped:
        base = ungrouped.pop()
        base_hash = image_hashes[base]
        group = [base]
        to_remove = set()
        for other in ungrouped:
            if image_hashes[other] - base_hash <= hash_threshold:
                group.append(other)
                to_remove.add(other)
        ungrouped -= to_remove
        similarity_groups.append(group)
    # Add images with no hash to their own group
    for img_name, h in image_hashes.items():
        if h is None:
            similarity_groups.append([img_name])
    logging.info(f"Formed {len(similarity_groups)} similarity groups.")
    # --- Populate grid by similarity group ---
    row = 0
    col_count = 5
    for group_idx, group in enumerate(similarity_groups):
        group_hash = str(image_hashes[group[0]]) if image_hashes[group[0]] is not None else "NO_HASH"
        logging.info(f"Placing group {group_idx+1}/{len(similarity_groups)}: hash {group_hash} with {len(group)} images")
        group_label = QLabel(f"Similarity group: {group_hash}")
        group_label.setStyleSheet("color: #3daee9; background: #232629; font-weight: bold; font-size: 12pt;")
        grid.addWidget(group_label, row, 0, 1, col_count)
        row += 1
        col = 0
        for img_idx, img_name in enumerate(group):
            logging.debug(f"Loading image {img_name} in group {group_hash}")
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
            # Top label
            blur_score = compute_blur_score(img_path)
            if blur_score is not None:
                top_label = QLabel(f"Blur: {blur_score:.1f}")
            else:
                top_label = QLabel("")
            top_label.setStyleSheet("color: red; background: #222;")
            # Bottom label
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            hash_str = str(image_hashes[img_name]) if image_hashes[img_name] is not None else "NO_HASH"
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: {hash_str}")
            bottom_label.setStyleSheet("color: white; background: #222;")
            # Blur/metrics label
            blur_label = QLabel("")
            blur_label.setStyleSheet("color: yellow; background: #222;")
            idx = len(image_labels)
            def on_enter(event, idx=idx):
                show_metrics(idx)
            def on_leave(event, idx=idx):
                hide_metrics(idx)
            lbl = HoverLabel(on_enter=on_enter, on_leave=on_leave)
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            lbl.setStyleSheet("background: #444; border: 2px solid #444;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
            grid.addWidget(lbl, row*4, col)
            grid.addWidget(top_label, row*4+1, col)
            grid.addWidget(bottom_label, row*4+2, col)
            grid.addWidget(blur_label, row*4+3, col)
            image_labels.append(lbl)
            top_labels.append(top_label)
            bottom_labels.append(bottom_label)
            blur_labels.append(blur_label)
            col += 1
            if col == col_count:
                col = 0
                row += 1
        row += 2
    status.showMessage(f"Loaded {num_images} images")
    win.show()
    app.exec()
    def on_close(event):
        app.setStyleSheet("")  # Reset style to default
        app.quit()
    win.closeEvent = on_close

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

def compute_perceptual_hash(img_path):
    try:
        img = Image.open(img_path)
        return imagehash.phash(img)
    except Exception as e:
        logging.warning(f"Could not compute perceptual hash for {img_path}: {e}")
        return None

class HoverLabel(QLabel):
    def __init__(self, *args, on_enter=None, on_leave=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_enter = on_enter
        self._on_leave = on_leave
    def enterEvent(self, event):
        if self._on_enter:
            self._on_enter(event)
        super().enterEvent(event)
    def leaveEvent(self, event):
        if self._on_leave:
            self._on_leave(event)
        super().leaveEvent(event)
