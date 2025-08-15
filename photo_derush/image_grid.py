from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea, QVBoxLayout
from .widgets import HoverEffectLabel
from .utils import pil2pixmap, compute_blur_score, compute_sharpness_features

class ImageGrid(QWidget):
    def __init__(self, image_paths, directory, info_panel, status_bar, get_sorted_images, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = image_paths
        self.directory = directory
        self.info_panel = info_panel
        self.status_bar = status_bar
        self.get_sorted_images = get_sorted_images
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
        self.populate_grid()

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
            top_label = QLabel("")
            top_label.setStyleSheet("color: red; background: #222;")
            date_str = str(os.path.getmtime(img_path)) if os.path.exists(img_path) else "N/A"
            bottom_label = QLabel(f"{img_name}\nDate: {date_str}\nHash: ...")
            bottom_label.setStyleSheet("color: white; background: #222;")
            blur_label = QLabel("")
            blur_label.setStyleSheet("color: yellow; background: #222;")
            lbl = HoverEffectLabel()
            lbl.setPixmap(pix)
            lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
            lbl.setAlignment(lbl.alignment())
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
                    self.info_panel.update_info(img_name, img_path, "-", "...", "...", metrics)
                return handler
            lbl.mousePressEvent = mousePressEventFactory(idx, lbl, img_name, img_path)
            self.grid.addWidget(lbl, (idx//self.col_count)*4, idx%self.col_count)
            self.grid.addWidget(top_label, (idx//self.col_count)*4+1, idx%self.col_count)
            self.grid.addWidget(bottom_label, (idx//self.col_count)*4+2, idx%self.col_count)
            self.grid.addWidget(blur_label, (idx//self.col_count)*4+3, idx%self.col_count)
            self.image_labels.append(lbl)
            self.top_labels.append(top_label)
            self.bottom_labels.append(bottom_label)
            self.blur_labels.append(blur_label)
            self.image_name_to_widgets[img_name] = (lbl, top_label, bottom_label, blur_label)
        self.status_bar.showMessage(f"Loaded {num_images} images (thumbnails only, grouping pending)")

