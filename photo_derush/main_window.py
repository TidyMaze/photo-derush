from PySide6.QtWidgets import QMainWindow, QStatusBar, QSplitter, QWidget, QApplication
from .toolbar import SettingsToolbar
from .info_panel import InfoPanel
from .image_grid import ImageGrid
from .viewer import open_full_image_qt
import os
import json
from ml.features import feature_vector
from ml.personal_learner import PersonalLearner
from ml.persistence import save_model, append_event, rebuild_model_from_log, clear_model_and_log, iter_events
import logging

class LightroomMainWindow(QMainWindow):
    def __init__(self, image_paths, directory, get_sorted_images, image_info=None):
        super().__init__()
        self.directory = directory  # ensure available before learner init
        self.setWindowTitle("Photo Derush (Qt)")
        self.resize(1400, 800)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.toolbar = SettingsToolbar(self)
        self.addToolBar(self.toolbar)
        self.info_panel = InfoPanel()
        self.splitter = QSplitter()
        self.setCentralWidget(self.splitter)
        self.left_panel = QWidget()
        self.sort_by_group = False
        self.image_info = image_info or {}
        def get_sorted_images():
            if self.sort_by_group and self.image_info:
                # Sort by group, then by filename
                def group_key(img):
                    info = self.image_info.get(img, {})
                    group = info.get("group")
                    return (group if group is not None else 999999, img)
                return sorted(image_paths, key=group_key)
            return image_paths
        self.current_img_idx = 0
        self.sorted_images = image_paths
        self.learner = None
        self._init_learner()
        # Create image grid once with callbacks
        self.image_grid = ImageGrid(image_paths, directory, self.info_panel, self.status, get_sorted_images,
                                    image_info=image_info, on_open_fullscreen=self.open_fullscreen,
                                    on_select=self.on_select_image)
        self.splitter.addWidget(self.image_grid)
        self.splitter.addWidget(self.info_panel)
        self.splitter.setSizes([1000, 400])
        self.toolbar.zoom_changed.connect(self.image_grid.set_cell_size)
        self.toolbar.sort_by_group_action.toggled.connect(self.on_sort_by_group_toggled)
        self.toolbar.keep_clicked.connect(self.on_keep_clicked)
        self.toolbar.trash_clicked.connect(self.on_trash_clicked)
        self.toolbar.unsure_clicked.connect(self.on_unsure_clicked)
        self.toolbar.predict_sort_clicked.connect(self.on_predict_sort_clicked)
        self.toolbar.export_csv_clicked.connect(self.on_export_csv_clicked)
        self.toolbar.reset_model_clicked.connect(self.on_reset_model_clicked)
        self.logger = logging.getLogger(__name__)

    def _init_learner(self):
        # Use first image to get feature size
        if self.sorted_images:
            fv, _ = feature_vector(os.path.join(self.directory, self.sorted_images[0]))
            self.learner = PersonalLearner(n_features=len(fv))
            rebuild_model_from_log(self.learner)
        else:
            self.learner = None

    def get_current_image(self):
        if 0 <= self.current_img_idx < len(self.sorted_images):
            return self.sorted_images[self.current_img_idx]
        return None

    def on_keep_clicked(self):
        self._label_current_image(1)
    def on_trash_clicked(self):
        self._label_current_image(0)
    def on_unsure_clicked(self):
        self._label_current_image(-1)

    def _label_current_image(self, label):
        img_name = self.get_current_image()
        if img_name is None:
            return
        img_path = os.path.join(self.directory, img_name)
        fv_tuple = feature_vector(img_path)
        if fv_tuple is None:
            import logging as _logging
            _logging.warning("[Learner] Feature extraction failed for %s; skipping label", img_path)
            return
        fv, _ = fv_tuple
        self.logger.info("[Label] User set label=%s for image=%s", label, img_name)
        event = {'image': img_name, 'path': img_path, 'features': fv.tolist(), 'label': label}
        append_event(event)
        if label in (0, 1):
            self.logger.debug("[Training] Starting incremental update for image=%s label=%s", img_name, label)
            self.learner.partial_fit([fv], [label])
            save_model(self.learner)
            self.logger.debug("[Training] Model saved after update")
        self.refresh_keep_prob()

    def refresh_keep_prob(self):
        img_name = self.get_current_image()
        if img_name is None:
            return
        img_path = os.path.join(self.directory, img_name)
        fv, _ = feature_vector(img_path)
        if fv is None:
            self.logger.warning("[Predict] Feature extraction failed for %s; cannot predict", img_path)
            return
        self.logger.debug("[Predict] Computing keep probability for image=%s", img_name)
        keep_prob = float(self.learner.predict_keep_prob([fv])[0]) if fv is not None else None
        self.logger.info("[Predict] keep_prob=%.4f image=%s", keep_prob, img_name)
        self.info_panel.update_info(img_name, img_path, "-", "-", "-", keep_prob=keep_prob)

    def on_predict_sort_clicked(self):
        # Sort remaining images by keep probability (desc)
        scored = []
        for img_name in self.sorted_images:
            img_path = os.path.join(self.directory, img_name)
            fv, _ = feature_vector(img_path)
            prob = float(self.learner.predict_keep_prob([fv])[0]) if fv is not None else 0.5
            scored.append((prob, img_name))
        scored.sort(reverse=True)
        self.sorted_images = [img for _, img in scored]
        self.image_grid.image_paths = self.sorted_images
        self.image_grid.populate_grid()

    def on_export_csv_clicked(self):
        # Export all labeled images with keep_prob
        rows = []
        for event in iter_events():
            img_path = event['path']
            label = event['label']
            fv = event['features']
            prob = float(self.learner.predict_keep_prob([fv])[0]) if fv is not None else 0.5
            rows.append({'path': img_path, 'label': label, 'keep_prob': prob})
        with open('labels.csv', 'w') as f:
            f.write('path,label,keep_prob\n')
            for row in rows:
                f.write(f"{row['path']},{row['label']},{row['keep_prob']:.4f}\n")

    def on_reset_model_clicked(self):
        clear_model_and_log()
        self._init_learner()
        self.refresh_keep_prob()

    def on_sort_by_group_toggled(self, checked):
        self.sort_by_group = checked
        self.image_grid.populate_grid()
    def closeEvent(self, event):
        QApplication.quit()
        super().closeEvent(event)

    def on_select_image(self, idx):
        if 0 <= idx < len(self.sorted_images):
            self.current_img_idx = idx

    def open_fullscreen(self, idx, img_path):
        self.on_select_image(idx)
        def keep_cb():
            self._label_current_image(1)
        def trash_cb():
            self._label_current_image(0)
        def unsure_cb():
            self._label_current_image(-1)
        open_full_image_qt(img_path, on_keep=keep_cb, on_trash=trash_cb, on_unsure=unsure_cb)
