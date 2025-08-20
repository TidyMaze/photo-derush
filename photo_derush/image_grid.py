import logging
import datetime

from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QSizePolicy, QToolButton, QStackedLayout, QHBoxLayout, QGraphicsDropShadowEffect, QStyle, QProgressBar
from PySide6.QtGui import QIcon, QColor, QPixmap, QImage, QPainter, QMovie
from PySide6.QtCore import Qt, QSize, QObject, Signal, QRunnable, QThreadPool
from .widgets import HoverEffectLabel
from .utils import pil2pixmap, compute_blur_score, compute_sharpness_features
from .image_manager import image_manager
from .feature_cache import FeatureVectorCache
import sys
sys.path.append('..')
from ml.personal_learner import PersonalLearner
from ml.persistence import load_model
import os

# Lightroom-like light color (less yellow, more neutral gray)
LIGHTROOM_LIGHT = QColor(200, 200, 200)

class ThumbnailResultEmitter(QObject):
    finished = Signal(int, str, object, dict, str, object)  # idx, img_name, info, group_to_color, default_color, pil_thumb

class ThumbnailLoader(QRunnable):
    def __init__(self, idx, img_name, info, group_to_color, default_color, directory, thumb_size, image_manager, emitter):
        super().__init__()
        self.idx = idx
        self.img_name = img_name
        self.info = info
        self.group_to_color = group_to_color
        self.default_color = default_color
        self.directory = directory
        self.thumb_size = thumb_size
        self.image_manager = image_manager
        self.emitter = emitter
    def run(self):
        import os
        img_path = os.path.join(self.directory, self.img_name)
        pil_thumb = self.image_manager.get_thumbnail(img_path, (self.thumb_size, self.thumb_size))
        self.emitter.finished.emit(self.idx, self.img_name, self.info, self.group_to_color, self.default_color, pil_thumb)

class FeatureExtractionEmitter(QObject):
    finished = Signal(list)  # emits list of feature vectors
    progress = Signal(int, int)  # emits (completed, total)

class FeatureExtractionWorker(QRunnable):
    def __init__(self, feature_cache, img_paths, emitter):
        super().__init__()
        self.feature_cache = feature_cache
        self.img_paths = img_paths
        self.emitter = emitter

    def run(self):
        import threading
        logging.info(f"[ThreadCheck] FeatureExtractionWorker running in thread: {threading.current_thread().name}")
        results = []
        total = len(self.img_paths)
        completed = 0
        for img_path in self.img_paths:
            cached = self.feature_cache.get(img_path)
            if cached is not None:
                results.append(cached)
            else:
                try:
                    result = self.feature_cache.feature_vector_fn(img_path)
                    self.feature_cache.set(img_path, result)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Feature extraction failed for {img_path}: {e}")
                    self.feature_cache.set(img_path, None)
                    results.append(None)
            completed += 1
            self.emitter.progress.emit(completed, total)
        self.emitter.finished.emit(results)

class ImageGrid(QWidget):
    def __init__(self, image_paths, directory, info_panel, status_bar, get_sorted_images, image_info=None, on_open_fullscreen=None, on_select=None, labels_map=None, get_feature_vector_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = image_paths
        self.directory = directory
        self.info_panel = info_panel
        self.status_bar = status_bar
        self.get_sorted_images = get_sorted_images
        self.image_info = image_info or {}
        self.THUMB_SIZE = 160
        self.MAX_IMAGES = 400
        self.col_count = 5
        self.image_labels = []
        self.top_labels = []
        self.bottom_labels = []
        self.blur_labels = []
        self.metrics_cache = {}
        self.image_name_to_widgets = {}
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background-color: #23272e;")
        self.grid = QGridLayout(self.grid_container)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_container)
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.scroll)
        self.setLayout(layout)
        self.learner = None
        self.feature_keys = None
        self.on_open_fullscreen = on_open_fullscreen
        self.on_select = on_select
        self.labels_map = labels_map or {}
        self.base_bottom_texts = {}
        self._logged_keep_prob_images: set[str] = set()  # track images already logged for keep probability
        self._get_feature_vector_fn = get_feature_vector_fn
        # Feature vector cache for fast lookup and parallel extraction
        self._feature_cache = FeatureVectorCache(self._get_feature_vector_fn, max_workers=8)
        self._last_prob_map = {}
        self.selected_image_name = None  # Track the selected image name
        # Do not populate yet; caller may stream images
        if image_paths:
            self.populate_grid()
        self._init_learner()

        # Modern font and color palette for grid
        self.setStyleSheet("""
            QWidget {
                font-family: system-ui, Arial, sans-serif;
                font-size: 13px;
                color: #f0f0f0;
            }
        """)

    def _init_learner(self):
        # Try to load model, else create new
        sample_img = None
        for img in self.image_paths:
            sample_img = os.path.join(self.directory, img)
            break
        if isinstance(sample_img, (str, bytes, os.PathLike)) and os.path.exists(sample_img):
            fv_tuple = self._get_cached_feature_vector(sample_img)
            if fv_tuple:
                fv, keys = fv_tuple
                self.feature_keys = keys
                model = load_model()
                if model is not None:
                    self.learner = model
                else:
                    self.learner = PersonalLearner(n_features=len(fv))
            else:
                # Feature extraction failed; defer learner init
                self.learner = None
        else:
            self.learner = None

    def clear_grid(self):
        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

    def populate_grid(self):
        self.clear_grid()
        self.image_labels.clear(); self.top_labels.clear(); self.bottom_labels.clear(); self.blur_labels.clear(); self.image_name_to_widgets.clear()
        sorted_images = self.get_sorted_images()
        num_images = min(self.MAX_IMAGES, len(sorted_images))
        img_paths = [os.path.join(self.directory, img) for img in sorted_images[:num_images]]
        # Start batch extraction in background (non-blocking) using QRunnable
        self._feature_extraction_emitter = FeatureExtractionEmitter()
        self._feature_extraction_emitter.finished.connect(self._on_feature_extraction_done)
        self._feature_extraction_emitter.progress.connect(self._on_feature_extraction_progress)
        self._feature_threadpool = QThreadPool.globalInstance()
        feature_worker = FeatureExtractionWorker(self._feature_cache, img_paths, self._feature_extraction_emitter)
        self._feature_threadpool.start(feature_worker)
        import colorsys
        group_to_color = {}
        used_groups = {self.image_info.get(img, {}).get('group') for img in sorted_images[:num_images]}
        used_groups = sorted(g for g in used_groups if g is not None)
        n = max(1, len(used_groups))
        palette = []
        for i in range(n):
            hue = i / n; rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)
            palette.append('#%02x%02x%02x' % tuple(int(255*x) for x in rgb))
        for idx, g in enumerate(used_groups):
            group_to_color[g] = palette[idx % len(palette)]
        default_color = '#444444'
        # Threaded loading
        self._pending_thumbs = set()
        self._thumb_emitter = ThumbnailResultEmitter()
        self._thumb_emitter.finished.connect(self._on_thumb_ready)
        self._thumb_threadpool = QThreadPool.globalInstance()
        for idx, img_name in enumerate(sorted_images[:num_images]):
            info = self.image_info.get(img_name, {})
            loader = ThumbnailLoader(idx, img_name, info, group_to_color, default_color, self.directory, self.THUMB_SIZE, image_manager, self._thumb_emitter)
            self._pending_thumbs.add(img_name)
            self._thumb_threadpool.start(loader)

    def _get_cached_feature_vector(self, img_path):
        fv = self._feature_cache.get(img_path)
        if fv is not None:
            return fv
        # Fallback to on-demand extraction if not yet cached
        return self._get_feature_vector_fn(img_path) if self._get_feature_vector_fn else None

    def _on_thumb_ready(self, idx, img_name, info, group_to_color, default_color, pil_thumb):
        if pil_thumb is None:
            return
        # Only add if still pending (avoid race if grid was cleared)
        if not hasattr(self, '_pending_thumbs') or img_name not in self._pending_thumbs:
            return
        self._pending_thumbs.remove(img_name)
        group = info.get('group')
        color = group_to_color.get(group, default_color)
        hash_str = info.get('hash', '...')
        group_str = group if group is not None else '...'
        self._add_thumbnail_row(img_name, idx, color=color, hash_str=hash_str, group_str=group_str, pil_thumb=pil_thumb)
        self.status_bar.showMessage(f"Loaded {len(self.image_labels)} images (thumbnails only, grouping pending)")
        # After all thumbnails are loaded, restore selection and info panel
        if not self._pending_thumbs:
            if self.selected_image_name in self.image_name_to_widgets:
                lbl, _, _, _ = self.image_name_to_widgets[self.selected_image_name]
                for l in self.image_labels:
                    if isinstance(l, HoverEffectLabel):
                        l.set_selected(False)
                lbl.set_selected(True)
                # Update info panel for the selected image
                img_path = os.path.join(self.directory, self.selected_image_name)
                info = self.image_info.get(self.selected_image_name, {})
                hash_str = info.get('hash', '...')
                group = info.get('group')
                group_str = group if group is not None else '...'
                blur_score = compute_blur_score(img_path)
                sharpness_metrics = compute_sharpness_features(img_path)
                aesthetic_score = 42
                metrics = (blur_score, sharpness_metrics, aesthetic_score)
                keep_prob = None
                if self.learner is not None:
                    img_path = os.path.join(self.directory, self.selected_image_name)
                    fv_tuple = self._get_cached_feature_vector(img_path)
                    if fv_tuple is not None:
                        fv, _ = fv_tuple
                        keep_prob = float(self.learner.predict_keep_prob([fv])[0])
                self.info_panel.update_info(self.selected_image_name, img_path, "-", hash_str, group_str, metrics, keep_prob=keep_prob)
            else:
                self.selected_image_name = None
                self.info_panel.update_info("", "", "", "", "", (), keep_prob=None)

    def _on_feature_extraction_done(self, results):
        import threading
        logging.info(f"[ThreadCheck] _on_feature_extraction_done running in thread: {threading.current_thread().name}")
        logging.info(f"Feature extraction completed for {len(results)} images.")
        self.progress_bar.hide()
        # You can update the UI or cache here if needed

    def _on_feature_extraction_progress(self, completed, total):
        import threading
        logging.info(f"[ThreadCheck] _on_feature_extraction_progress running in thread: {threading.current_thread().name}")
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
            if completed >= total:
                self.progress_bar.hide()

    def _add_thumbnail_row(self, img_name, idx, color='#444444', hash_str='...', group_str='...', pil_thumb=None):
        import os
        from PySide6.QtGui import QColor
        import logging
        img_path = os.path.join(self.directory, img_name)
        # Loading spinner placeholder
        spinner = QLabel()
        spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        spinner.setStyleSheet("background: #23272e; border-radius: 8px;")
        spinner_movie = QMovie(":/qt-project.org/styles/commonstyle/images/working-32.gif")
        spinner.setMovie(spinner_movie)
        spinner_movie.start()
        # Add spinner to grid while loading
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)
        container.setStyleSheet("""
            background: #282c34;
            border-radius: 12px;
            border: 1px solid #333;
            /* transition: box-shadow 0.2s; Removed unsupported property */
        """)
        try:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(12)
            shadow.setOffset(0, 2)
            shadow.setColor(QColor("gray"))
            container.setGraphicsEffect(shadow)
        except Exception:
            pass
        vbox.addStretch(1)
        vbox.addWidget(spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox.addStretch(1)
        row = idx // self.col_count
        col = idx % self.col_count
        self.grid.addWidget(container, row, col)
        self.grid.setColumnMinimumWidth(col, self.THUMB_SIZE + 20)
        container.setFixedWidth(self.THUMB_SIZE + 20)
        container.setFixedHeight(self.THUMB_SIZE + 60)
        # After thumbnail is ready, replace spinner with actual content
        if pil_thumb is None:
            pil_thumb = image_manager.get_thumbnail(img_path, (self.THUMB_SIZE, self.THUMB_SIZE))
        if pil_thumb is None:
            # Show error icon if image fails to load
            error_label = QLabel()
            error_label.setPixmap(QIcon.fromTheme("dialog-error").pixmap(self.THUMB_SIZE, self.THUMB_SIZE))
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setToolTip("Failed to load image")
            vbox.addWidget(error_label, alignment=Qt.AlignmentFlag.AlignCenter)
            return
        # Remove spinner
        spinner.hide()
        pix = pil2pixmap(pil_thumb)
        group_badge = group_str if group_str not in (None, '', '...', 'None') else ''
        # Compute date_str before using it
        date_str = "N/A"
        if img_path and isinstance(img_path, (str, bytes)) and os.path.exists(img_path):
            try:
                ts = os.path.getmtime(img_path)
                date_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logging.warning(f"Could not format date for {img_path}: {e}")
        bottom_label = QLabel(f"{img_name}\nDate: {date_str}")
        self.base_bottom_texts[img_name] = bottom_label.text()
        bottom_label.setStyleSheet(f"color: {LIGHTROOM_LIGHT.name()}; background: transparent; font-size: 11px;")
        # Use icons for label, now as a larger, more rounded badge
        lbl_val = self.labels_map.get(img_name)
        badge_color = "#888"
        badge_icon = QIcon()
        badge_tooltip = "Unlabeled"
        icon_size = 20
        def make_icon(icon, color):
            if not icon.isNull():
                pixmap = icon.pixmap(icon_size, icon_size)
                img = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                painter = QPainter(img)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(img.rect(), QColor(color))
                painter.end()
                pm = QPixmap.fromImage(img)
                pm.setDevicePixelRatio(self.devicePixelRatioF())
                return QIcon(pm)
            return icon
        if lbl_val == 1:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            badge_color = "#2e7d32"
            badge_icon = make_icon(icon, "white")
            badge_tooltip = "Keep"
        elif lbl_val == 0:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
            badge_color = "#b71c1c"
            badge_icon = make_icon(icon, "white")
            badge_tooltip = "Trash"
        elif lbl_val == -1:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
            badge_color = "#fbc02d"
            badge_icon = make_icon(icon, "black")
            badge_tooltip = "Unsure"
        else:
            # Fallback for unknown label
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
            badge_color = "#888"
            badge_icon = make_icon(icon, "black")
            badge_tooltip = "Unlabeled"
        import logging
        overlay = OverlayWidget(group_badge, badge_icon, badge_color, badge_tooltip)
        overlay.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        # Image label
        lbl = HoverEffectLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Connect click/double-click to selection/fullscreen
        lbl.clicked.connect(lambda name=img_name: self.on_thumbnail_clicked(name))
        lbl.doubleClicked.connect(lambda name=img_name: self.on_thumbnail_double_clicked(name))
        # Stack image and overlay
        stack = QStackedLayout()
        stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        img_container = QWidget()
        img_container.setLayout(stack)
        stack.addWidget(lbl)
        stack.addWidget(overlay)
        vbox.addWidget(img_container, alignment=Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(bottom_label, alignment=Qt.AlignmentFlag.AlignCenter)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.image_labels.append(lbl)
        self.top_labels.append(overlay.group_label)
        self.bottom_labels.append(bottom_label)
        self.blur_labels.append(overlay.badge)
        self.image_name_to_widgets[img_name] = (lbl, overlay.group_label, bottom_label, overlay.badge)
        # Update status bar with progress
        total_images = min(self.MAX_IMAGES, len(self.get_sorted_images()))
        loaded_images = len(self.image_labels)
        self.status_bar.showMessage(f"Loaded {loaded_images}/{total_images} images (thumbnails only, grouping pending)")

    def set_cell_size(self, size):
        self.THUMB_SIZE = size
        self.populate_grid()

    def update_blur_label(self, img_name, blur_label):
        lbl_val = self.labels_map.get(img_name)
        badge_color = "#888"
        badge_icon = QIcon()
        badge_tooltip = "Unlabeled"
        icon_size = 20
        def make_icon(icon, color):
            if not icon.isNull():
                pixmap = icon.pixmap(icon_size, icon_size)
                img = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                painter = QPainter(img)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(img.rect(), QColor(color))
                painter.end()
                pm = QPixmap.fromImage(img)
                pm.setDevicePixelRatio(self.devicePixelRatioF())
                return QIcon(pm)
            return icon
        if lbl_val == 1:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
            badge_color = "#2e7d32"
            badge_icon = make_icon(icon, "white")
            badge_tooltip = "Keep"
        elif lbl_val == 0:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
            badge_color = "#b71c1c"
            badge_icon = make_icon(icon, "white")
            badge_tooltip = "Trash"
        elif lbl_val == -1:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
            badge_color = "#fbc02d"
            badge_icon = make_icon(icon, "black")
            badge_tooltip = "Unsure"
        # Find the badge QToolButton and update it
        widgets = self.image_name_to_widgets.get(img_name)
        if widgets:
            badge_btn = widgets[3]
            badge_btn.setIcon(badge_icon)
            badge_btn.setStyleSheet(f"border-radius: 14px; background: {badge_color}; border: 2px solid white;")
            badge_btn.setToolTip(badge_tooltip)

    def update_label(self, img_name, label):
        self.labels_map[img_name] = label
        widgets = self.image_name_to_widgets.get(img_name)
        if not widgets:
            return
        blur_label = widgets[3]
        self.update_blur_label(img_name, blur_label)

    def add_image(self, img_name: str):
        logging.info("[ImageGrid] Adding image: %s", img_name)
        if img_name in self.image_name_to_widgets or hasattr(self, '_pending_thumbs') and img_name in self._pending_thumbs:
            return
        if len(self.image_labels) >= self.MAX_IMAGES:
            return
        idx = len(self.image_labels)
        # Append for consistency in get_sorted_images
        if img_name not in self.image_paths:
            self.image_paths.append(img_name)
        info = self.image_info.get(img_name, {})
        color = '#444444'
        # Ensure emitter and threadpool are initialized
        if not hasattr(self, '_thumb_emitter'):
            self._thumb_emitter = ThumbnailResultEmitter()
            self._thumb_emitter.finished.connect(self._on_thumb_ready)
        if not hasattr(self, '_thumb_threadpool'):
            self._thumb_threadpool = QThreadPool.globalInstance()
        if not hasattr(self, '_pending_thumbs'):
            self._pending_thumbs = set()
        group_to_color = {info.get('group'): color} if info.get('group') is not None else {}
        default_color = color
        loader = ThumbnailLoader(idx, img_name, info, group_to_color, default_color, self.directory, self.THUMB_SIZE, image_manager, self._thumb_emitter)
        self._pending_thumbs.add(img_name)
        self._thumb_threadpool.start(loader)
        self.status_bar.showMessage(f"Loading image: {img_name} (streaming)")

    def update_keep_probabilities(self, prob_map: dict):
        """Append or update keep probability line in bottom labels.
        Logs keep probability only once per image (first time it's set).
        Now displays probability with gradient color (red->yellow->green) and colored border.
        """
        self._last_prob_map = prob_map or {}
        def _interp_color(p: float):
            # clamp
            p = max(0.0, min(1.0, p))
            # 0 -> red (#b71c1c), 0.5 -> amber (#ffa000), 1 -> green (#2e7d32)
            if p < 0.5:
                # red to amber
                t = p / 0.5
                c0 = (0xb7,0x1c,0x1c)
                c1 = (0xff,0xa0,0x00)
            else:
                t = (p - 0.5) / 0.5
                c0 = (0xff,0xa0,0x00)
                c1 = (0x2e,0x7d,0x32)
            r = int(c0[0] + (c1[0]-c0[0])*t)
            g = int(c0[1] + (c1[1]-c0[1])*t)
            b = int(c0[2] + (c1[2]-c0[2])*t)
            return f"#{r:02x}{g:02x}{b:02x}"
        for img_name, widgets in self.image_name_to_widgets.items():
            lbl, _, bottom_label, _ = widgets
            # Retrieve original base text (stored when row created)
            raw_text = bottom_label.text()
            if img_name in self.base_bottom_texts:
                base = self.base_bottom_texts[img_name]
            else:
                if '<span' in raw_text and 'Prob:' in raw_text:
                    parts = raw_text.split('<br>')
                    if parts and 'Prob:' in parts[-1]:
                        base = '\n'.join(p for p in parts[:-1])
                    else:
                        base = raw_text
                else:
                    base = raw_text.split('\nProb:')[0]
            prob = prob_map.get(img_name)
            if prob is not None:
                pct = prob * 100.0
                col = _interp_color(prob)
                base_html = base.replace('\n', '<br>')
                prob_html = f"<span style=\"background-color:{col}; color:#e6e6dc; font-weight:bold; padding:1px 4px; border-radius:3px;\">Prob: {pct:.0f}%</span>"
                bottom_label.setText(f"{base_html}<br>{prob_html}")
                # Border color to reinforce probability
                try:
                    lbl.setStyleSheet(f"border: 3px solid {col};")
                except Exception:
                    pass
                if img_name not in self._logged_keep_prob_images:
                    logging.info("[ImageGrid] Updated keep probability for %s: %.2f (%.0f%%)", img_name, prob, pct)
                    self._logged_keep_prob_images.add(img_name)
            else:
                bottom_label.setText(base)
                try:
                    lbl.setStyleSheet("")
                except Exception:
                    pass

    def resizeEvent(self, event):
        # Responsive grid: adjust column count based on width
        width = self.scroll.viewport().width()
        new_col_count = max(1, width // (self.THUMB_SIZE + 32))
        if new_col_count != self.col_count:
            self.col_count = new_col_count
            self.populate_grid()
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        # Keyboard navigation in grid
        if not self.image_labels:
            return super().keyPressEvent(event)
        idx = 0
        if self.selected_image_name and self.selected_image_name in self.image_name_to_widgets:
            idx = self.image_labels.index(self.image_name_to_widgets[self.selected_image_name][0])
        key = event.key()
        if key == Qt.Key.Key_Left:
            idx = max(0, idx - 1)
        elif key == Qt.Key.Key_Right:
            idx = min(len(self.image_labels) - 1, idx + 1)
        elif key == Qt.Key.Key_Up:
            idx = max(0, idx - self.col_count)
        elif key == Qt.Key.Key_Down:
            idx = min(len(self.image_labels) - 1, idx + self.col_count)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.on_open_fullscreen and self.selected_image_name:
                img_path = os.path.join(self.directory, self.selected_image_name)
                self.on_open_fullscreen(idx, img_path)
                return
        else:
            return super().keyPressEvent(event)
        # Update selection
        lbl = self.image_labels[idx]
        for l in self.image_labels:
            if hasattr(l, 'set_selected'):
                l.set_selected(False)
        lbl.set_selected(True)
        # Update info panel
        img_name = None
        for name, widgets in self.image_name_to_widgets.items():
            if widgets[0] == lbl:
                img_name = name
                break
        if img_name:
            self.selected_image_name = img_name
            img_path = os.path.join(self.directory, img_name)
            info = self.image_info.get(img_name, {})
            hash_str = info.get('hash', '...')
            group = info.get('group')
            group_str = group if group is not None else '...'
            blur_score = compute_blur_score(img_path)
            sharpness_metrics = compute_sharpness_features(img_path)
            aesthetic_score = 42
            metrics = (blur_score, sharpness_metrics, aesthetic_score)
            keep_prob = None
            if self.learner is not None:
                img_path = os.path.join(self.directory, self.selected_image_name)
                fv_tuple = self._get_cached_feature_vector(img_path)
                if fv_tuple is not None:
                    fv, _ = fv_tuple
                    keep_prob = float(self.learner.predict_keep_prob([fv])[0])
            self.info_panel.update_info(img_name, img_path, "-", hash_str, group_str, metrics, keep_prob=keep_prob)

    def on_thumbnail_clicked(self, img_name):
        # Deselect all
        for l in self.image_labels:
            if hasattr(l, 'set_selected'):
                l.set_selected(False)
        # Select clicked
        widgets = self.image_name_to_widgets.get(img_name)
        if widgets:
            lbl = widgets[0]
            if hasattr(lbl, 'set_selected'):
                lbl.set_selected(True)
            self.selected_image_name = img_name
            # Update info panel
            img_path = os.path.join(self.directory, img_name)
            info = self.image_info.get(img_name, {})
            hash_str = info.get('hash', '...')
            group = info.get('group')
            group_str = group if group is not None else '...'
            blur_score = compute_blur_score(img_path)
            sharpness_metrics = compute_sharpness_features(img_path)
            aesthetic_score = 42
            metrics = (blur_score, sharpness_metrics, aesthetic_score)
            keep_prob = None
            if self.learner is not None:
                fv_tuple = self._get_feature_vector_fn(img_path) if self._get_feature_vector_fn else None
                if fv_tuple is not None:
                    fv, _ = fv_tuple
                    keep_prob = float(self.learner.predict_keep_prob([fv])[0])
            self.info_panel.update_info(img_name, img_path, "-", hash_str, group_str, metrics, keep_prob=keep_prob)

    def on_thumbnail_double_clicked(self, img_name):
        import logging
        logging.info(f"Double-clicked: {img_name}")
        if self.on_open_fullscreen:
            sorted_images = self.get_sorted_images()
            idx = sorted_images.index(img_name)
            img_path = os.path.join(self.directory, img_name)
            self.on_open_fullscreen(idx, img_path, sorted_images)

class OverlayWidget(QWidget):
    def __init__(self, group_text, badge_icon, badge_color, badge_tooltip, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Group label (top-left)
        safe_group_text = str(group_text) if group_text is not None else ''
        self.group_label = QLabel(safe_group_text)
        self.group_label.setStyleSheet("background: rgba(0,0,0,0.6); color: white; border-radius: 8px; padding: 2px 6px; font-size: 12px;")
        self.group_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.group_label, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Spacer
        layout.addStretch(1)
        # Badge (top-right)
        self.badge = QToolButton()
        self.badge.setIcon(badge_icon)
        self.badge.setIconSize(QSize(20, 20))
        self.badge.setFixedSize(28, 28)
        self.badge.setStyleSheet(f"border-radius: 14px; background: {badge_color}; border: 2px solid white;")
        self.badge.setToolTip(badge_tooltip)
        self.badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.badge, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

    def update_badge(self, icon, color, tooltip):
        self.badge.setIcon(icon)
        self.badge.setStyleSheet(f"border-radius: 14px; background: {color}; border: 2px solid white;")
        self.badge.setToolTip(tooltip)

    def update_group(self, text):
        self.group_label.setText(str(text) if text is not None else '')
