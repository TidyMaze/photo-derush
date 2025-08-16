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
from ml.persistence import load_model
# import helpers for latest-only logic
from ml.persistence import latest_labeled_events, load_latest_labeled_samples
import logging
import numpy as np

class LightroomMainWindow(QMainWindow):
    def __init__(self, image_paths, directory, get_sorted_images, image_info=None):
        super().__init__()
        self.directory = directory
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
        self.current_img_idx = 0
        self.sorted_images = image_paths
        self.learner = None
        self.labels_map = self._build_labels_map_from_log()
        self.logger = logging.getLogger(__name__)
        # Cold-start training gate (require min samples per class before training)
        self._cold_start_completed = False
        # Do not init learner if no images yet
        if self.sorted_images:
            self._init_learner()
        # create grid
        self.image_grid = ImageGrid(self.sorted_images, directory, self.info_panel, self.status, self._compute_sorted_images,
                                    image_info=self.image_info, on_open_fullscreen=self.open_fullscreen,
                                    on_select=self.on_select_image, labels_map=self.labels_map)
        # Share learner instance with grid (avoid separate untrained model)
        self.image_grid.learner = self.learner
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

    def _compute_sorted_images(self):
        if self.sort_by_group and self.image_info:
            def group_key(img):
                info = self.image_info.get(img, {})
                group = info.get("group")
                return (group if group is not None else 999999, img)
            return sorted(self.sorted_images, key=group_key)
        return self.sorted_images

    def load_images(self, image_paths, image_info):
        self.logger.info("[AsyncLoad] Applying prepared images: %d", len(image_paths))
        self.image_info = image_info or {}
        self.sorted_images = image_paths
        if self.image_grid is None:
            self.image_grid = ImageGrid(self.sorted_images, self.directory, self.info_panel, self.status, self._compute_sorted_images,
                                        image_info=self.image_info, on_open_fullscreen=self.open_fullscreen,
                                        on_select=self.on_select_image, labels_map=self.labels_map)
            # Share learner instance with grid (avoid separate untrained model)
            self.image_grid.learner = self.learner
            self.splitter.insertWidget(0, self.image_grid)
            self.splitter.setSizes([1000, 400])
        else:
            self.image_grid.image_paths = self.sorted_images
            self.image_grid.image_info = self.image_info
            self.image_grid.populate_grid()
        try:
            count = len(getattr(self.image_grid, 'image_labels', []))
            self.logger.info("[AsyncLoad] Grid populated with %d thumbnails", count)
            self.status.showMessage(f"Loaded {count} images")
        except Exception as e:
            self.logger.warning("[AsyncLoad] Could not determine grid count: %s", e)
        if self.learner is None and self.sorted_images:
            self._init_learner()
        self.refresh_keep_prob()

    def _definitive_label_counts(self):
        counts = {0: 0, 1: 0}
        latest = latest_labeled_events()
        for ev in latest.values():
            lbl = ev.get('label')
            if lbl in (0,1):
                counts[lbl] += 1
        return counts

    def _ensure_learner(self, fv=None):
        """Guarantee self.learner is initialized.
        Order of attempts:
          1. If already initialized -> return.
          2. If saved model exists -> load it (assume already trained) and mark cold start complete.
          3. If provided fv (definitive label path) -> init with its length (UNTRAINED yet) and maybe rebuild if threshold met.
          4. Scan event log for a definitive labeled sample (0/1) -> infer n_features & maybe rebuild if threshold met.
          5. Derive feature length from first available image (no training yet).
        Training (rebuild from log) only occurs if both classes have >= MIN_CLASS_SAMPLES.
        """
        if self.learner is not None:
            return
        MIN_CLASS_SAMPLES = 10
        counts = self._definitive_label_counts()
        # Try load persisted model first
        try:
            persisted = load_model()
        except Exception as e:  # noqa: PERF203
            persisted = None
            self.logger.warning("[Learner] Failed to load persisted model: %s", e)
        if persisted is not None:
            self.learner = persisted
            self._cold_start_completed = True
            if hasattr(self, 'image_grid'):
                self.image_grid.learner = self.learner
            self.logger.info("[Learner] Loaded persisted model (cold start satisfied)")
            return
        # Provided fresh fv
        if fv is not None:
            self.logger.info("[Learner] Initializing (untrained) from provided feature vector (%d dims)", len(fv))
            self.learner = PersonalLearner(n_features=len(fv))
            if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                self.logger.info("[Learner] Cold-start threshold met (%d/%d); rebuilding from log", counts[0], counts[1])
                rebuild_model_from_log(self.learner)
                self._cold_start_completed = True
            else:
                self.logger.info("[Learner] Cold-start threshold not yet met (class0=%d class1=%d need %d each) -> postponing training", counts[0], counts[1], MIN_CLASS_SAMPLES)
            if hasattr(self, 'image_grid'):
                self.image_grid.learner = self.learner
            return
        # Rebuild from log path (only if threshold satisfied)
        if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
            # Need feature length guess from first qualifying latest event
            latest = latest_labeled_events()
            for ev in latest.values():
                feats = ev.get('features')
                if feats:
                    self.logger.info("[Learner] Initializing from latest log replay (%d dims) after threshold met", len(feats))
                    self.learner = PersonalLearner(n_features=len(feats))
                    rebuild_model_from_log(self.learner)
                    self._cold_start_completed = True
                    if hasattr(self, 'image_grid'):
                        self.image_grid.learner = self.learner
                    return
        # Derive from first image if possible (untrained)
        if self.sorted_images:
            first_path = os.path.join(self.directory, self.sorted_images[0])
            fv_tuple = feature_vector(first_path)
            if fv_tuple is not None:
                fv0, _ = fv_tuple
                self.logger.info("[Learner] Initializing size from first image only (%d dims, untrained; waiting for threshold)", len(fv0))
                self.learner = PersonalLearner(n_features=len(fv0))
                if hasattr(self, 'image_grid'):
                    self.image_grid.learner = self.learner

    def _init_learner(self):
        # Backwards-compat entry point – now just ensures learner.
        if self.sorted_images:
            self._ensure_learner()

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

    def _build_labels_map_from_log(self):
        labels = {}
        latest = latest_labeled_events()
        for img, ev in latest.items():
            lbl = ev.get('label')
            if lbl is not None:
                labels[os.path.basename(img)] = lbl
        return labels

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
        self.labels_map[img_name] = label
        # Ensure learner only for definitive labels
        if label in (0, 1):
            self._ensure_learner(fv)
            if self.learner is not None:
                # Check cold start status
                counts = self._definitive_label_counts()
                MIN_CLASS_SAMPLES = 10
                if not self._cold_start_completed:
                    if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                        # Perform initial training over all historical data
                        self.logger.info("[Training] Cold-start threshold reached (class0=%d class1=%d). Performing initial full training.", counts[0], counts[1])
                        rebuild_model_from_log(self.learner)
                        self._cold_start_completed = True
                        save_model(self.learner)
                        self.logger.info("[Training] Initial full training complete and model saved")
                    else:
                        self.logger.info("[Training] Skipping update (cold-start threshold not met: class0=%d class1=%d need %d each)", counts[0], counts[1], MIN_CLASS_SAMPLES)
                if self._cold_start_completed:
                    # Incremental update with just the new sample
                    self.logger.info("[Training] Incremental update for image=%s label=%s", img_name, label)
                    self.learner.partial_fit([fv], [label])
                    save_model(self.learner)
                    self.logger.info("[Training] Model saved after incremental update")
                    # Evaluate after update
                    self._evaluate_model()
        # update UI badge
        if hasattr(self, 'image_grid'):
            self.image_grid.update_label(img_name, label)
        if self.learner is not None:
            # Single image refresh for info panel
            self.refresh_keep_prob()
            # Batch refresh all visible thumbs (always refresh after any label change)
            self._refresh_all_keep_probs()
            logging.info("[Label] Updated keep probabilities for all images in grid after labeling %s", img_name)

    def _evaluate_model(self):
        """Evaluate current learner on latest definitive labeled events (0/1) only."""
        if self.learner is None:
            return
        X, y, _images = load_latest_labeled_samples()
        if not X:
            return
        import numpy as _np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
        X_np = _np.asarray(X, dtype=_np.float64)
        y_np = _np.asarray(y, dtype=_np.int64)
        try:
            probs = self.learner.predict_keep_prob(X_np)
            preds = (probs >= 0.5).astype(int)
            pos_rate = float(y_np.mean())
            acc = float(accuracy_score(y_np, preds))
            try:
                prec = float(precision_score(y_np, preds, zero_division=0))
                rec = float(recall_score(y_np, preds, zero_division=0))
                f1 = float(f1_score(y_np, preds, zero_division=0))
            except Exception:
                prec = rec = f1 = 0.0
            try:
                ll = float(log_loss(y_np, _np.vstack([1-probs, probs]).T, labels=[0,1]))
            except Exception:
                ll = float('nan')
            try:
                brier = float(brier_score_loss(y_np, probs))
            except Exception:
                brier = float('nan')
            self.logger.info("[Eval] samples=%d pos_rate=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f logloss=%s brier=%s (latest labels only)",
                             len(y_np), pos_rate, acc, prec, rec, f1,
                             f"{ll:.4f}" if _np.isfinite(ll) else "nan",
                             f"{brier:.4f}" if _np.isfinite(brier) else "nan")
        except Exception as e:
            self.logger.warning("[Eval] Failed evaluation: %s", e)

    def _refresh_all_keep_probs(self):
        if self.learner is None or not self.image_grid:
            return
        names = []
        vectors = []
        # Only need probabilities for images currently in grid (respect MAX_IMAGES)
        for img_name in self.image_grid.image_paths[:self.image_grid.MAX_IMAGES]:
            img_path = os.path.join(self.directory, img_name)
            fv_tuple = feature_vector(img_path)
            if fv_tuple is None:
                logging.warning("[Predict] Feature extraction failed for %s; skipping", img_path)
                continue

            logging.info("[Predict] Extracting feature vector for image=%s", img_name)

            fv, _ = fv_tuple
            names.append(img_name)
            vectors.append(fv)
        if not vectors:
            return
        try:
            logging.info("[Predict] Predicting keep probabilities for %d images", len(vectors))
            probs = self.learner.predict_keep_prob(vectors)
            logging.info("[Predict] Computed keep probabilities for %d images", len(probs))
            prob_map = {n: float(p) for n, p in zip(names, probs)}
            self.image_grid.update_keep_probabilities(prob_map)
            logging.info("Done updating proba in grid for %d images", len(prob_map))
        except Exception as e:
            self.logger.warning("[Predict] Failed batch probability refresh: %s", e)

    def refresh_keep_prob(self):
        img_name = self.get_current_image()
        if img_name is None:
            return
        self._ensure_learner()  # May initialize (untrained) or rebuild from log
        if self.learner is None:
            self.logger.info("[Predict] Skipping keep_prob refresh (no learner)")
            return
        img_path = os.path.join(self.directory, img_name)
        fv_tuple = feature_vector(img_path)
        if fv_tuple is None:
            self.logger.warning("[Predict] Feature extraction failed for %s; cannot predict", img_path)
            return
        fv, _ = fv_tuple
        self.logger.info("[Predict] Computing keep probability for image=%s", img_name)
        keep_prob = float(self.learner.predict_keep_prob([fv])[0]) if fv is not None else None
        self.logger.info("[Predict] keep_prob=%.4f image=%s", keep_prob, img_name)
        self.info_panel.update_info(img_name, img_path, "-", "-", "-", keep_prob=keep_prob)
        logging.info("[Predict] Updated keep_prob for image=%s: %.4f", img_name, keep_prob)

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
        # Export latest labeled images with keep_prob
        rows = []
        latest = latest_labeled_events()
        for img, ev in latest.items():
            img_path = ev.get('path') or img
            label = ev.get('label')
            feats = ev.get('features')
            prob = float(self.learner.predict_keep_prob([feats])[0]) if (self.learner is not None and feats is not None) else 0.5
            rows.append({'path': img_path, 'label': label, 'keep_prob': prob})
        with open('labels.csv', 'w') as f:
            f.write('path,label,keep_prob\n')
            for row in rows:
                f.write(f"{row['path']},{row['label']},{row['keep_prob']:.4f}\n")

    def on_reset_model_clicked(self):
        # Preserve event log; only delete persisted model so we can start fresh while keeping history.
        clear_model_and_log(delete_log=False)
        self.logger.info("="*60)
        self.logger.info("[Reset] Cleared persisted model ONLY (log preserved). Starting brand-new untrained learner with fresh scaler.")
        self.logger.info("="*60)
        # New empty learner (do NOT rebuild from log to keep it truly fresh)
        self.learner = None
        # Clear in-memory labels so UI badges reflect fresh state (log still holds history)
        self.labels_map = {}
        if hasattr(self, 'image_grid'):
            self.image_grid.labels_map = self.labels_map
            self.image_grid.populate_grid()
        # Determine feature dimension from first image (if any)
        if self.sorted_images:
            first_path = os.path.join(self.directory, self.sorted_images[0])
            fv_tuple = feature_vector(first_path)
            if fv_tuple is not None:
                fv0, _ = fv_tuple
                self.learner = PersonalLearner(n_features=len(fv0))
                if hasattr(self, 'image_grid'):
                    self.image_grid.learner = self.learner
        self.refresh_keep_prob()

    def on_sort_by_group_toggled(self, checked):
        self.sort_by_group = checked
        if self.image_grid:
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

    def update_grouping(self, image_info):
        self.logger.info("[AsyncLoad] Updating grouping metadata for %d images", len(image_info))
        self.image_info = image_info or {}
        if self.image_grid:
            self.image_grid.image_info = self.image_info
            if self.sort_by_group:
                self.sorted_images = self._compute_sorted_images()
            self.image_grid.populate_grid()
