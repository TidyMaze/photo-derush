import logging

from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool
from .widgets import HoverEffectLabel
from .utils import pil2pixmap, compute_blur_score, compute_sharpness_features
from .image_manager import image_manager
import sys
sys.path.append('..')
from ml.personal_learner import PersonalLearner
from ml.persistence import load_model, save_model
import os

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
        self.grid_container.setStyleSheet("background-color: #222;")
        self.grid = QGridLayout(self.grid_container)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_container)
        layout = QVBoxLayout(self)
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
        self._last_prob_map = {}
        self.selected_image_name = None  # Track the selected image name
        # Do not populate yet; caller may stream images
        if image_paths:
            self.populate_grid()
        self._init_learner()

    def _init_learner(self):
        # Try to load model, else create new
        sample_img = None
        for img in self.image_paths:
            sample_img = os.path.join(self.directory, img)
            break
        if sample_img and os.path.exists(sample_img):
            fv_tuple = self._get_feature_vector_fn(sample_img) if self._get_feature_vector_fn else None
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
                    fv_tuple = self._get_feature_vector_fn(img_path) if self._get_feature_vector_fn else None
                    if fv_tuple is not None:
                        fv, _ = fv_tuple
                        keep_prob = float(self.learner.predict_keep_prob([fv])[0])
                self.info_panel.update_info(self.selected_image_name, img_path, "-", hash_str, group_str, metrics, keep_prob=keep_prob)
            else:
                self.selected_image_name = None
                self.info_panel.update_info("", "", "", "", "", (), keep_prob=None)

    def _add_thumbnail_row(self, img_name, idx, color='#444444', hash_str='...', group_str='...', pil_thumb=None):
        import os
        img_path = os.path.join(self.directory, img_name)
        if pil_thumb is None:
            pil_thumb = image_manager.get_thumbnail(img_path, (self.THUMB_SIZE, self.THUMB_SIZE))
        if pil_thumb is None:
            return  # skip if cannot load
        pix = pil2pixmap(pil_thumb)
        group_badge = group_str if group_str not in (None, '', '...', 'None') else ''
        top_label = QLabel(str(group_badge))
        top_label.setStyleSheet(f"background: {color}; color: #fff; font-weight: bold; border-radius: 8px; min-height: 18px; padding: 2px 8px; text-align: center;")
        date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
        bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: {hash_str}")
        self.base_bottom_texts[img_name] = bottom_label.text()
        bottom_label.setStyleSheet("color: white; background: #222;")
        blur_label = QLabel("")
        lbl_val = self.labels_map.get(img_name)
        if lbl_val == 1:
            blur_label.setText("KEEP"); blur_label.setStyleSheet("color: #fff; background:#2e7d32; font-weight:bold; padding:2px;")
        elif lbl_val == 0:
            blur_label.setText("TRASH"); blur_label.setStyleSheet("color: #fff; background:#b71c1c; font-weight:bold; padding:2px;")
        elif lbl_val == -1:
            blur_label.setText("UNSURE"); blur_label.setStyleSheet("color: #000; background:#ffeb3b; font-weight:bold; padding:2px;")
        else:
            blur_label.setStyleSheet("color: yellow; background: #222;")
        lbl = HoverEffectLabel()
        lbl.setPixmap(pix)
        lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path, hash_str=hash_str, group_str=group_str):
            def handler(e):
                for l in self.image_labels:
                    if isinstance(l, HoverEffectLabel):
                        l.set_selected(False)
                label.set_selected(True)
                self.selected_image_name = img_name  # Track the selected image
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
                if self.on_select:
                    self.on_select(idx)
            return handler
        def mouseDoubleClickEventFactory(idx=idx, img_path=img_path):
            def handler(e):
                if self.on_open_fullscreen:
                    self.on_open_fullscreen(idx, img_path)
            return handler
        lbl.mousePressEvent = mousePressEventFactory()
        lbl.mouseDoubleClickEvent = mouseDoubleClickEventFactory()
        # --- NEW GRID LAYOUT ---
        # Use globally imported QVBoxLayout, QWidget, QSizePolicy
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(4)
        vbox.addWidget(top_label)
        lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        vbox.addWidget(lbl)
        vbox.addWidget(bottom_label)
        vbox.addWidget(blur_label)
        container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row = idx // self.col_count
        col = idx % self.col_count
        self.grid.addWidget(container, row, col)
        self.image_labels.append(lbl)
        self.top_labels.append(top_label)
        self.bottom_labels.append(bottom_label)
        self.blur_labels.append(blur_label)
        self.image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)

    def set_cell_size(self, size):
        self.THUMB_SIZE = size
        self.populate_grid()

    def update_label(self, img_name, label):
        self.labels_map[img_name] = label
        widgets = self.image_name_to_widgets.get(img_name)
        if not widgets:
            return
        _, _, _, blur_label = widgets
        if label == 1:
            blur_label.setText("KEEP")
            blur_label.setStyleSheet("color: #fff; background:#2e7d32; font-weight:bold; padding:2px;")
        elif label == 0:
            blur_label.setText("TRASH")
            blur_label.setStyleSheet("color: #fff; background:#b71c1c; font-weight:bold; padding:2px;")
        elif label == -1:
            blur_label.setText("UNSURE")
            blur_label.setStyleSheet("color: #000; background:#ffeb3b; font-weight:bold; padding:2px;")
        else:
            blur_label.setText("")
            blur_label.setStyleSheet("color: yellow; background: #222;")

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
            import math
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
                prob_html = f"<span style=\"background-color:{col}; color:#fff; font-weight:bold; padding:1px 4px; border-radius:3px;\">Prob: {pct:.0f}%</span>"
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
