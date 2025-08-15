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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QAction
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
    app.lastWindowClosed.connect(app.quit)
    qss_path = os.path.join(os.path.dirname(__file__), "qdarkstyle.qss")
    with open(qss_path, "r") as f:
        app.setStyleSheet(f.read())
    logging.info("QDarkStyle stylesheet applied.")
    win = LightroomMainWindow()
    win.setWindowTitle("Photo Derush (Qt)")
    win.resize(1400, 800)
    status = QStatusBar()
    win.setStatusBar(status)

    # --- Top toolbar with settings ---
    from PySide6.QtWidgets import QToolBar, QCheckBox, QHBoxLayout, QPushButton
    toolbar = QToolBar("Settings Toolbar")
    win.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
    sort_by_group_action = QAction("Sort by group", win)
    sort_by_group_action.setCheckable(True)
    toolbar.addAction(sort_by_group_action)
    # Store clusters and image_hashes for sorting
    clusters = []
    image_hashes = {}
    sort_by_group = False

    splitter = QSplitter()
    win.setCentralWidget(splitter)
    left_panel = QWidget()
    left_layout = QVBoxLayout(left_panel)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    left_layout.addWidget(scroll)
    grid_container = QWidget()
    grid_container.setStyleSheet("background-color: #222;")
    grid = QGridLayout(grid_container)
    scroll.setWidget(grid_container)
    right_panel = QWidget()
    right_panel.setFixedWidth(400)
    right_layout = QVBoxLayout(right_panel)
    info_label = QLabel(("This is the right panel.\n" * 50).strip())
    info_label.setStyleSheet("color: #aaa; background: #222; font-size: 14pt;")
    info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    info_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
    )
    right_layout.addWidget(info_label)
    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setSizes([1000, 400])
    logging.info("About to show window...")
    win.show()
    logging.info("Window shown, populating images...")

    # --- Helper to clear and repopulate the grid ---
    def clear_grid():
        for i in reversed(range(grid.count())):
            item = grid.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    # --- Helper to get sorted image list ---
    def get_sorted_images():
        if sort_by_group_action.isChecked() and clusters:
            # Flatten clusters, then add any images not in clusters
            clustered = [img for group in clusters for img in group]
            rest = [img for img in image_paths[:num_images] if img not in clustered]
            return clustered + rest
        else:
            return image_paths[:num_images]

    # --- Image grid population: show images immediately, no grouping ---
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

    def populate_grid():
        clear_grid()
        image_labels.clear()
        top_labels.clear()
        bottom_labels.clear()
        blur_labels.clear()
        image_name_to_widgets.clear()
        sorted_images = get_sorted_images()
        for idx, img_name in enumerate(sorted_images):
            img_path = os.path.join(directory, img_name)
            thumb_path = os.path.join(directory, 'thumbnails', img_name)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            if not os.path.exists(thumb_path):
                img = Image.open(img_path)
                img.thumbnail((THUMB_SIZE, THUMB_SIZE))
                img.save(thumb_path)
            img = Image.open(thumb_path)
            pix = pil2pixmap(img)
            top_label = QLabel("")
            top_label.setStyleSheet("color: red; background: #222;")
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: ...")
            bottom_label.setStyleSheet("color: white; background: #222;")
            blur_label = QLabel("")
            blur_label.setStyleSheet("color: yellow; background: #222;")
            lbl = HoverEffectLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path):
                def handler(e: QMouseEvent):
                    for l in image_labels:
                        if isinstance(l, HoverEffectLabel):
                            l.set_selected(False)
                    label.set_selected(True)
                    # Compute metrics and update right panel
                    blur_score = compute_blur_score(img_path)
                    sharpness_metrics = compute_sharpness_features(img_path)
                    aesthetic_score = 42
                    metrics = (blur_score, sharpness_metrics, aesthetic_score)
                    update_info_panel(img_name, img_path, "-", "...", "...", metrics)
                return handler
            lbl.mousePressEvent = mousePressEventFactory(idx, lbl, img_name, img_path)
            grid.addWidget(lbl, (idx//col_count)*4, idx%col_count)
            grid.addWidget(top_label, (idx//col_count)*4+1, idx%col_count)
            grid.addWidget(bottom_label, (idx//col_count)*4+2, idx%col_count)
            grid.addWidget(blur_label, (idx//col_count)*4+3, idx%col_count)
            image_labels.append(lbl)
            top_labels.append(top_label)
            bottom_labels.append(bottom_label)
            blur_labels.append(blur_label)
            image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)
        status.showMessage(f"Loaded {len(sorted_images)} images (thumbnails only, grouping pending)")

    populate_grid()

    def update_grid_with_groups(new_clusters, new_image_hashes):
        nonlocal clusters, image_hashes
        clusters = new_clusters
        image_hashes = new_image_hashes
        # Use a set of visually distinct colors for group highlighting
        group_colors = [
            "#FF6666", "#66FF66", "#6699FF", "#FFCC66", "#CC66FF", "#66FFFF", "#FF99CC",
            "#CCCCCC", "#FFB366", "#B3FF66", "#66B3FF", "#FF66B3", "#B366FF"
        ]
        image_to_group = {}
        for group_idx, group in enumerate(clusters):
            for img_name in group:
                image_to_group[img_name] = group_idx
        for img_name, widgets in image_name_to_widgets.items():
            lbl, top_label, bottom_label, blur_label = widgets
            group_idx = image_to_group.get(img_name)
            group_color = group_colors[group_idx % len(group_colors)] if group_idx is not None else "#444"
            if group_idx is not None:
                top_label.setText(f"Group {group_idx+1}")
                top_label.setStyleSheet(f"color: white; background: {group_color}; font-weight: bold;")
            else:
                top_label.setText("")
                top_label.setStyleSheet("color: red; background: #222;")
            lbl.setStyleSheet(f"background: #444; border: 2px solid {group_color};")
            img_hash = image_hashes.get(img_name, "...")
            lines = bottom_label.text().split("\n")
            if len(lines) >= 3:
                lines[2] = f"Hash: {img_hash}"
            bottom_label.setText("\n".join(lines))
        logging.info(f"Updated grid with {len(clusters)} groups.")

    def deferred_hashing_and_population():
        logging.info("Deferred hashing started (should be after window is visible)...")
        from main import cluster_duplicates
        logging.info("Clustering duplicates...")
        new_clusters, new_image_hashes = cluster_duplicates(image_paths, directory)
        logging.info(f"Clustering complete: {len(new_clusters)} groups found.")
        update_grid_with_groups(new_clusters, new_image_hashes)
        if on_window_opened:
            on_window_opened(new_clusters, new_image_hashes)
        status.showMessage(f"Grouping complete: {len(new_clusters)} groups")
    QTimer.singleShot(0, deferred_hashing_and_population)

    def update_info_panel(img_name, img_path, group_idx, group_hash, image_hash, metrics=None):
        exif = extract_exif(img_path)
        exif_lines = []
        for k, v in exif.items():
            if k == "GPSInfo" and isinstance(v, dict):
                exif_lines.append(f"GPSInfo: {format_gps_info(v)}")
            else:
                exif_lines.append(f"{k}: {v}")
        exif_str = "\n".join(exif_lines) if exif_lines else "No EXIF data"
        # Metrics
        metrics_str = ""
        if metrics:
            blur_score, sharpness_metrics, aesthetic_score = metrics
            lines = [
                f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A",
                f"Laplacian: {sharpness_metrics['variance_laplacian']:.1f}" if sharpness_metrics else "Laplacian: N/A",
                f"Tenengrad: {sharpness_metrics['tenengrad']:.1f}" if sharpness_metrics else "Tenengrad: N/A",
                f"Brenner: {sharpness_metrics['brenner']:.1f}" if sharpness_metrics else "Brenner: N/A",
                f"Wavelet: {sharpness_metrics['wavelet_energy']:.1f}" if sharpness_metrics else "Wavelet: N/A",
                f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A"
            ]
            metrics_str = "<b>Metrics:</b><br>" + "<br>".join(lines) + "<br>"
        info = f"<b>File:</b> {img_name}<br>"
        info += f"<b>Path:</b> {img_path}<br>"
        info += f"<b>Group ID:</b> {group_idx}<br>"
        info += f"<b>Group Hash:</b> {group_hash}<br>"
        info += f"<b>Image Hash:</b> {image_hash}<br>"
        info += metrics_str
        info += f"<b>EXIF:</b><br><pre style='font-size:10pt'>{exif_str}</pre>"
        info_label.setText(info)

    # --- React to sort by group toggle ---
    def on_sort_by_group_toggled():
        populate_grid()
        # If groups are available, update group coloring and hash
        if clusters and image_hashes:
            update_grid_with_groups(clusters, image_hashes)
    sort_by_group_action.toggled.connect(on_sort_by_group_toggled)

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

class HoverEffectLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_selected = False
        self._base_style = "background: #444; border: 2px solid #444;"
        self._hover_style = "background: #666; border: 2px solid #888;"
        self.setStyleSheet(self._base_style)
    def set_selected(self, selected: bool):
        self._is_selected = selected
        if selected:
            self.setStyleSheet("background: red; border: 2px solid red;")
        else:
            self.setStyleSheet(self._base_style)
    def enterEvent(self, event):
        if not self._is_selected:
            self.setStyleSheet(self._hover_style)
        super().enterEvent(event)
    def leaveEvent(self, event):
        if not self._is_selected:
            self.setStyleSheet(self._base_style)
        super().leaveEvent(event)
