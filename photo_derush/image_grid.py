import logging

from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
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

    def _add_thumbnail_row(self, img_name, idx, color='#444444', hash_str='...', group_str='...'):
        """Create thumbnail UI row (image + 3 labels) for given image name."""
        import os
        img_path = os.path.join(self.directory, img_name)
        # Use ImageManager to get/create thumbnail (cached in-memory + persisted)
        pil_thumb = image_manager.get_thumbnail(img_path, (self.THUMB_SIZE, self.THUMB_SIZE))
        if pil_thumb is None:
            return  # skip if cannot load
        pix = pil2pixmap(pil_thumb)
        top_label = QLabel("")
        top_label.setStyleSheet(f"background: {color}; min-height: 8px;")
        date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
        bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: {hash_str}\nGroup: {group_str}")
        self.base_bottom_texts[img_name] = bottom_label.text()
        bottom_label.setStyleSheet("color: white; background: #222;")
        blur_label = QLabel("")
        # Badge / label state
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
        lbl.setMinimumSize(self.THUMB_SIZE, self.THUMB_SIZE)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path, hash_str=hash_str, group_str=group_str):
            def handler(e):
                for l in self.image_labels:
                    if isinstance(l, HoverEffectLabel):
                        l.set_selected(False)
                label.set_selected(True)
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
        row_base = (idx // self.col_count) * 4
        col = idx % self.col_count
        self.grid.addWidget(lbl, row_base, col, alignment=Qt.AlignmentFlag.AlignCenter)
        self.grid.addWidget(top_label, row_base + 1, col)
        self.grid.addWidget(bottom_label, row_base + 2, col)
        self.grid.addWidget(blur_label, row_base + 3, col)
        self.image_labels.append(lbl)
        self.top_labels.append(top_label)
        self.bottom_labels.append(bottom_label)
        self.blur_labels.append(blur_label)
        self.image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)

    def populate_grid(self):
        self.clear_grid()
        self.image_labels.clear(); self.top_labels.clear(); self.bottom_labels.clear(); self.blur_labels.clear(); self.image_name_to_widgets.clear()
        sorted_images = self.get_sorted_images()
        num_images = min(self.MAX_IMAGES, len(sorted_images))
        import random, colorsys
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
        for idx, img_name in enumerate(sorted_images[:num_images]):
            info = self.image_info.get(img_name, {})
            group = info.get('group')
            color = group_to_color.get(group, default_color)
            hash_str = info.get('hash', '...')
            group_str = group if group is not None else '...'
            self._add_thumbnail_row(img_name, idx, color=color, hash_str=hash_str, group_str=group_str)
        self.status_bar.showMessage(f"Loaded {num_images} images (thumbnails only, grouping pending)")
        # Re-apply last known probabilities if available
        if self._last_prob_map:
            self.update_keep_probabilities(self._last_prob_map)

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
        if img_name in self.image_name_to_widgets:
            return
        if len(self.image_labels) >= self.MAX_IMAGES:
            return
        idx = len(self.image_labels)
        # Append for consistency in get_sorted_images
        if img_name not in self.image_paths:
            self.image_paths.append(img_name)
        # Use existing metadata if present else defaults
        info = self.image_info.get(img_name, {})
        color = '#444444'
        self._add_thumbnail_row(img_name, idx, color=color, hash_str=info.get('hash', '...'), group_str=str(info.get('group', '...')))
        self.status_bar.showMessage(f"Loaded {len(self.image_labels)} images (streaming)")

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
