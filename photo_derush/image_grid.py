from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
from .widgets import HoverEffectLabel
from .utils import pil2pixmap, compute_blur_score, compute_sharpness_features
import sys
sys.path.append('..')
from ml.features import feature_vector
from ml.personal_learner import PersonalLearner
from ml.persistence import load_model, save_model
import os

class ImageGrid(QWidget):
    def __init__(self, image_paths, directory, info_panel, status_bar, get_sorted_images, image_info=None, on_open_fullscreen=None, on_select=None, labels_map=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = image_paths
        self.directory = directory
        self.info_panel = info_panel
        self.status_bar = status_bar
        self.get_sorted_images = get_sorted_images
        self.image_info = image_info or {}
        self.THUMB_SIZE = 160
        self.MAX_IMAGES = 200
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
        if sample_img:
            fv, keys = feature_vector(sample_img)
            self.feature_keys = keys
            model = load_model()
            if model is not None:
                self.learner = model
            else:
                self.learner = PersonalLearner(n_features=len(fv))
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
        self.image_labels.clear()
        self.top_labels.clear()
        self.bottom_labels.clear()
        self.blur_labels.clear()
        self.image_name_to_widgets.clear()
        sorted_images = self.get_sorted_images()
        num_images = min(self.MAX_IMAGES, len(sorted_images))
        # Assign a color to each group
        import random
        import colorsys
        group_to_color = {}
        used_groups = set()
        for img_name in sorted_images[:num_images]:
            info = self.image_info.get(img_name, {})
            group = info.get("group", None)
            if group is not None:
                used_groups.add(group)
        used_groups = sorted(g for g in used_groups if g is not None)
        palette = []
        n = max(1, len(used_groups))
        for i in range(n):
            hue = i / n
            lightness = 0.5
            saturation = 0.7
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            color = '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)
            palette.append(color)
        for idx, group in enumerate(used_groups):
            group_to_color[group] = palette[idx % len(palette)]
        default_color = '#444444'
        for idx, img_name in enumerate(sorted_images[:num_images]):
            img_path = self.directory + '/' + img_name
            thumb_path = self.directory + '/thumbnails/' + img_name
            import os
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            if not os.path.exists(thumb_path):
                from PIL import Image
                img = Image.open(img_path)
                img.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE))
                img.save(thumb_path)
            from PIL import Image
            img = Image.open(thumb_path)
            pix = pil2pixmap(img)
            info = self.image_info.get(img_name, {})
            group = info.get("group", None)
            color = group_to_color.get(group, default_color)
            top_label = QLabel("")
            top_label.setStyleSheet(f"background: {color}; min-height: 8px;")
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            hash_str = info.get("hash", "...")
            group_str = group if group is not None else "..."
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: {hash_str}\nGroup: {group_str}")
            bottom_label.setStyleSheet("color: white; background: #222;")
            blur_label = QLabel("")
            # Set label badge if known
            lbl_val = self.labels_map.get(img_name)
            if lbl_val is not None:
                if lbl_val == 1:
                    blur_label.setText("KEEP")
                    blur_label.setStyleSheet("color: #fff; background:#2e7d32; font-weight:bold; padding:2px;")
                elif lbl_val == 0:
                    blur_label.setText("TRASH")
                    blur_label.setStyleSheet("color: #fff; background:#b71c1c; font-weight:bold; padding:2px;")
                elif lbl_val == -1:
                    blur_label.setText("UNSURE")
                    blur_label.setStyleSheet("color: #000; background:#ffeb3b; font-weight:bold; padding:2px;")
                else:
                    blur_label.setStyleSheet("color: yellow; background: #222;")
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
                    # --- Feature extraction and keep probability ---
                    keep_prob = None
                    if self.learner is not None:
                        fv, _ = feature_vector(img_path)
                        if fv is not None:
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
            lbl.mousePressEvent = mousePressEventFactory(idx, lbl, img_name, img_path, hash_str, group_str)
            lbl.mouseDoubleClickEvent = mouseDoubleClickEventFactory(idx, img_path)
            self.grid.addWidget(lbl, (idx//self.col_count)*4, idx%self.col_count, alignment=Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(top_label, (idx//self.col_count)*4+1, idx%self.col_count)
            self.grid.addWidget(bottom_label, (idx//self.col_count)*4+2, idx%self.col_count)
            self.grid.addWidget(blur_label, (idx//self.col_count)*4+3, idx%self.col_count)
            self.image_labels.append(lbl)
            self.top_labels.append(top_label)
            self.bottom_labels.append(bottom_label)
            self.blur_labels.append(blur_label)
            self.image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)
        self.status_bar.showMessage(f"Loaded {num_images} images (thumbnails only, grouping pending)")

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

    def add_image(self, img_name:str):
        if img_name in self.image_name_to_widgets:
            return
        if len(self.image_labels) >= self.MAX_IMAGES:
            return
        self.image_paths.append(img_name)
        idx = len(self.image_labels)
        import os
        img_path = os.path.join(self.directory, img_name)
        thumb_path = os.path.join(self.directory, 'thumbnails', img_name)
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        try:
            if not os.path.exists(thumb_path):
                from PIL import Image
                img = Image.open(img_path)
                img.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE))
                img.save(thumb_path)
            from PIL import Image
            img = Image.open(thumb_path)
        except Exception:
            return
        pix = pil2pixmap(img)
        group = None
        color = '#444444'
        top_label = QLabel("")
        top_label.setStyleSheet(f"background: {color}; min-height: 8px;")
        date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
        bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: ...\nGroup: ...")
        bottom_label.setStyleSheet("color: white; background: #222;")
        blur_label = QLabel("")
        lbl_val = self.labels_map.get(img_name)
        if lbl_val is not None:
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
        def mousePressEventFactory(idx=idx, label=lbl, img_name=img_name, img_path=img_path):
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
                    fv, _ = feature_vector(img_path)
                    if fv is not None:
                        keep_prob = float(self.learner.predict_keep_prob([fv])[0])
                self.info_panel.update_info(img_name, img_path, "-", "-", "-", metrics, keep_prob=keep_prob)
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
        # layout position
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
        self.status_bar.showMessage(f"Loaded {len(self.image_labels)} images (streaming)")
