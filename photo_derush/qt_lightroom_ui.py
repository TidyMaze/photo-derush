"""
PySide6 port of the Lightroom UI from main.py (Tkinter version).
- Main window: QMainWindow with left (image grid) and right (info) panels
- Image grid: QScrollArea + QGridLayout, thumbnails, selection, metrics overlay
- Full image viewer: QDialog or QMainWindow, closes on click/ESC
- All event handling and image display is Qt idiomatic
"""
import os
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QLabel, QGridLayout, QSplitter, QDialog, QStatusBar
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PIL import Image, ExifTags
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

class ClusteringWorker(QObject):
    finished = Signal(object, object)  # clusters, image_hashes
    def __init__(self, image_paths, directory):
        super().__init__()
        self.image_paths = image_paths
        self.directory = directory
    def run(self):
        from main import cluster_duplicates
        clusters, image_hashes = cluster_duplicates(self.image_paths, self.directory)
        self.finished.emit(clusters, image_hashes)

class ImageLoaderWorker(QObject):
    image_ready = Signal(str, object, object, object, object)  # img_name, pix, metrics, date_str, thumb_path
    finished = Signal()
    def __init__(self, image_paths, directory):
        super().__init__()
        self.image_paths = image_paths
        self.directory = directory
    def run(self):
        from PIL import Image
        import os
        for idx, img_name in enumerate(self.image_paths):
            img_path = os.path.join(self.directory, img_name)
            thumb_path = os.path.join(self.directory, 'thumbnails', img_name)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
            else:
                img = Image.open(img_path)
                img.thumbnail((160, 160))
                img.save(thumb_path)
            img.thumbnail((160, 160))
            from .qt_lightroom_ui import pil2pixmap, compute_blur_score, compute_sharpness_features
            pix = pil2pixmap(img)
            blur_score = compute_blur_score(img_path)
            sharpness_metrics = compute_sharpness_features(img_path)
            metrics = (blur_score, sharpness_metrics, 42)
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            self.image_ready.emit(img_name, pix, metrics, date_str, thumb_path)
        self.finished.emit()

class LightroomMainWindow(QMainWindow):
    def closeEvent(self, event):
        QApplication.quit()
        super().closeEvent(event)

# --- Main Lightroom UI ---
def show_lightroom_ui_qt(image_paths, directory, trashed_paths=None, trashed_dir=None, on_window_opened=None):
    logging.info("Starting Photo Derush Qt app...")
    logging.info(f"Image directory: {directory}")
    logging.info(f"Number of images: {len(image_paths)})")
    app = QApplication.instance() or QApplication([])
    # Ensure app quits when last window is closed
    app.lastWindowClosed.connect(app.quit)
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
    right_panel.setFixedWidth(400)
    right_layout = QVBoxLayout(right_panel)
    info_label = QLabel(("This is the right panel.\n" * 50).strip())
    info_label.setStyleSheet("color: #aaa; background: #222; font-size: 14pt;")
    info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse)
    right_layout.addWidget(info_label)
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setSizes([1000, 400])
    logging.info("About to show window...")
    win.show()
    logging.info("Window shown, scheduling deferred hashing...")
    if on_window_opened:
        QTimer.singleShot(0, on_window_opened)

    # --- Image grid population: PHASE 1 (show images immediately, no grouping) ---
    THUMB_SIZE = 160
    num_images = min(MAX_IMAGES, len(image_paths))
    image_labels = []
    top_labels = []
    bottom_labels = []
    blur_labels = []
    metrics_cache = {}
    image_name_to_widgets = {}
    row = 0
    col_count = 5
    col = 0
    for idx, img_name in enumerate(image_paths[:num_images]):
        img_path = os.path.join(directory, img_name)
        thumb_path = os.path.join(directory, 'thumbnails', img_name)
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        if os.path.exists(thumb_path):
            img = Image.open(thumb_path)
        else:
            img = Image.open(img_path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(thumb_path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        pix = pil2pixmap(img)
        # Top label (placeholder)
        top_label = QLabel("")
        top_label.setStyleSheet("color: red; background: #222;")
        # Bottom label (filename, no hash yet)
        date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
        bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: ...")
        bottom_label.setStyleSheet("color: white; background: #222;")
        # Blur/metrics label
        blur_label = QLabel("")
        blur_label.setStyleSheet("color: yellow; background: #222;")
        def show_metrics(event=None, idx=idx):
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
            blur_label.setText("\n".join(lines))
        def hide_metrics(event=None, idx=idx):
            blur_label.setText("")
        lbl = HoverLabel(on_enter=show_metrics, on_leave=hide_metrics)
        lbl.setPixmap(pix)
        lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        lbl.setStyleSheet("background: #444; border: 2px solid #444;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path):
            def handler(e: QMouseEvent):
                for l in image_labels:
                    l.setStyleSheet("background: #444; border: 2px solid #444;")
                label.setStyleSheet("background: red; border: 2px solid red;")
                update_info_panel(img_name, img_path, "-", "...", "...")
            return handler
        lbl.mousePressEvent = mousePressEventFactory(idx, lbl, img_name, img_path)
        grid.addWidget(lbl, row*4, col)
        grid.addWidget(top_label, row*4+1, col)
        grid.addWidget(bottom_label, row*4+2, col)
        grid.addWidget(blur_label, row*4+3, col)
        image_labels.append(lbl)
        top_labels.append(top_label)
        bottom_labels.append(bottom_label)
        blur_labels.append(blur_label)
        image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)
        col += 1
        if col == col_count:
            col = 0
            row += 1
    status.showMessage(f"Loaded {num_images} images (thumbnails only, grouping pending)")

    # --- PHASE 2: Deferred hashing/grouping, update UI when ready ---
    def deferred_hashing_and_population():
        logging.info("Deferred hashing started (should be after window is visible)...")
        from main import cluster_duplicates
        logging.info("Clustering duplicates...")
        clusters, image_hashes = cluster_duplicates(image_paths, directory)
        logging.info(f"Clustering complete: {len(clusters)} groups found.")
        if on_window_opened:
            on_window_opened(clusters, image_hashes)
        # --- Image grid population: PHASE 1 (show images immediately, no grouping) ---
        THUMB_SIZE = 160
        num_images = min(MAX_IMAGES, len(image_paths))
        image_labels = []
        top_labels = []
        bottom_labels = []
        blur_labels = []
        metrics_cache = {}
        # Store mapping for later group update
        image_name_to_widgets = {}
        row = 0
        col_count = 5
        col = 0
        for idx, img_name in enumerate(image_paths[:num_images]):
            img_path = os.path.join(directory, img_name)
            thumb_path = os.path.join(directory, 'thumbnails', img_name)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
            else:
                img = Image.open(img_path)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(thumb_path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            pix = pil2pixmap(img)
            # Top label (placeholder)
            top_label = QLabel("")
            top_label.setStyleSheet("color: red; background: #222;")
            # Bottom label (filename, no hash yet)
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: ...")
            bottom_label.setStyleSheet("color: white; background: #222;")
            # Blur/metrics label
            blur_label = QLabel("")
            blur_label.setStyleSheet("color: yellow; background: #222;")
            def show_metrics(event=None, idx=idx):
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
                blur_label.setText("\n".join(lines))
            def hide_metrics(event=None, idx=idx):
                blur_label.setText("")
            lbl = HoverLabel(on_enter=show_metrics, on_leave=hide_metrics)
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            lbl.setStyleSheet("background: #444; border: 2px solid #444;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path):
                def handler(e: QMouseEvent):
                    for l in image_labels:
                        l.setStyleSheet("background: #444; border: 2px solid #444;")
                    label.setStyleSheet("background: red; border: 2px solid red;")
                    update_info_panel(img_name, img_path, "-", "...", "...")
                return handler
            lbl.mousePressEvent = mousePressEventFactory(idx, lbl, img_name, img_path)
            grid.addWidget(lbl, row*4, col)
            grid.addWidget(top_label, row*4+1, col)
            grid.addWidget(bottom_label, row*4+2, col)
            grid.addWidget(blur_label, row*4+3, col)
            image_labels.append(lbl)
            top_labels.append(top_label)
            bottom_labels.append(bottom_label)
            blur_labels.append(blur_label)
            image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)
            col += 1
            if col == col_count:
                col = 0
                row += 1
        status.showMessage(f"Loaded {num_images} images (thumbnails only, grouping pending)")

        # --- PHASE 2: Deferred hashing/grouping, update UI when ready ---
        def update_grid_with_groups(clusters, image_hashes):
            # Remove all group labels if any
            for i in reversed(range(grid.count())):
                widget = grid.itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text().startswith("Similarity group:"):
                    grid.removeWidget(widget)
                    widget.deleteLater()
            # Add group labels and update hash info
            row = 0
            col_count = 5
            for group_idx, group in enumerate(clusters):
                group_hash = str(image_hashes[group[0]]) if image_hashes[group[0]] is not None else "NO_HASH"
                group_label = QLabel(f"Similarity group: {group_hash}")
                group_label.setStyleSheet("color: #3daee9; background: #232629; font-weight: bold; font-size: 12pt;")
                grid.addWidget(group_label, row, 0, 1, col_count)
                row += 1
                col = 0
                for img_name in group:
                    if img_name in image_name_to_widgets:
                        lbl, top_label, bottom_label, blur_label = image_name_to_widgets[img_name]
                        hash_str = str(image_hashes[img_name]) if image_hashes[img_name] is not None else "NO_HASH"
                        bottom_label.setText(f"{img_name}\nHash: {hash_str}")
                        # Update click handler to provide group info
                        def mousePressEventFactory(img_name=img_name, group_idx=group_idx, group_hash=group_hash, image_hash=hash_str, label=lbl):
                            def handler(e: QMouseEvent):
                                for l in image_labels:
                                    l.setStyleSheet("background: #444; border: 2px solid #444;")
                                label.setStyleSheet("background: red; border: 2px solid red;")
                                update_info_panel(img_name, os.path.join(directory, img_name), group_idx, group_hash, image_hash)
                            return handler
                        lbl.mousePressEvent = mousePressEventFactory(img_name, group_idx, group_hash, hash_str, lbl)
                        # Optionally, update other labels if needed
                        grid.addWidget(lbl, row*4, col)
                        grid.addWidget(top_label, row*4+1, col)
                        grid.addWidget(bottom_label, row*4+2, col)
                        grid.addWidget(blur_label, row*4+3, col)
                        col += 1
                        if col == col_count:
                            col = 0
                            row += 1
                row += 2
            status.showMessage(f"Grouping complete: {len(clusters)} groups")

        # Start clustering in background after UI is responsive
        def start_deferred_grouping():
            worker = ClusteringWorker(image_paths, directory)
            thread = QThread()
            worker.moveToThread(thread)
            def on_finished(clusters, image_hashes):
                update_grid_with_groups(clusters, image_hashes)
                if on_window_opened:
                    on_window_opened(clusters, image_hashes)
                thread.quit()
                worker.deleteLater()
                thread.deleteLater()
            worker.finished.connect(on_finished)
            thread.started.connect(worker.run)
            thread.start()
        QTimer.singleShot(0, start_deferred_grouping)
    QTimer.singleShot(0, deferred_hashing_and_population)

    def update_info_panel(img_name, img_path, group_idx, group_hash, image_hash):
        exif = extract_exif(img_path)
        exif_lines = []
        for k, v in exif.items():
            if k == "GPSInfo" and isinstance(v, dict):
                exif_lines.append(f"GPSInfo: {format_gps_info(v)}")
            else:
                exif_lines.append(f"{k}: {v}")
        exif_str = "\n".join(exif_lines) if exif_lines else "No EXIF data"
        info = f"<b>File:</b> {img_name}<br>"
        info += f"<b>Path:</b> {img_path}<br>"
        info += f"<b>Group ID:</b> {group_idx}<br>"
        info += f"<b>Group Hash:</b> {group_hash}<br>"
        info += f"<b>Image Hash:</b> {image_hash}<br>"
        info += f"<b>EXIF:</b><br><pre style='font-size:10pt'>{exif_str}</pre>"
        info_label.setText(info)

    app.exec()
    logging.info("Qt event loop exited. Application quitting.")
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

def extract_exif(img_path):
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if not exif_data:
            return {}
        exif = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            exif[decoded] = value
        return exif
    except Exception as e:
        logging.warning(f"Could not extract EXIF for {img_path}: {e}")
        return {}

def format_gps_info(gps_info):
    def _convert_to_degrees(value):
        d, m, s = value
        return float(d[0]) / float(d[1]) + \
               float(m[0]) / float(m[1]) / 60 + \
               float(s[0]) / float(s[1]) / 3600
    try:
        gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = lon = None
        if 'GPSLatitude' in gps_tags and 'GPSLatitudeRef' in gps_tags:
            lat = _convert_to_degrees(gps_tags['GPSLatitude'])
            if gps_tags['GPSLatitudeRef'] in ['S', b'S']:
                lat = -lat
        if 'GPSLongitude' in gps_tags and 'GPSLongitudeRef' in gps_tags:
            lon = _convert_to_degrees(gps_tags['GPSLongitude'])
            if gps_tags['GPSLongitudeRef'] in ['W', b'W']:
                lon = -lon
        if lat is not None and lon is not None:
            return f"Latitude: {lat:.6f}, Longitude: {lon:.6f}"
        return str(gps_tags)
    except Exception as e:
        return f"[Invalid GPSInfo: {e}]"

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
