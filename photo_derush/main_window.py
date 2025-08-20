from PySide6.QtWidgets import QMainWindow, QStatusBar, QSplitter, QWidget
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout
from .toolbar import SettingsToolbar
from .info_panel import InfoPanel
from .image_grid import ImageGrid
from .viewer import open_full_image_qt
import os
import json
from ml.features import feature_vector, all_feature_names  # added all_feature_names import
# Legacy alias for tests must be defined early so monkeypatching works
compute_feature_vector = feature_vector  # noqa: E305
try:
    __all__
except NameError:
    __all__ = []
if 'compute_feature_vector' not in __all__:
    __all__.append('compute_feature_vector')
from ml.features_cv import FEATURE_NAMES as _NEW_FEATURE_NAMES
# Backward compatibility: expose feature_vector symbol (tests may monkeypatch it)
from ml.personal_learner import PersonalLearner
from ml.persistence import save_model, append_event, rebuild_model_from_log, clear_model_and_log
from ml.persistence import load_model
from ml.persistence import load_feature_cache, persist_feature_cache_entry
# import helpers for latest-only logic
from ml.persistence import latest_labeled_events, load_latest_labeled_samples
import logging
import numpy as np
from PySide6.QtCore import QRunnable, QObject, Signal, QThreadPool
import time
from PySide6.QtGui import QFont, QAction, QIcon
from PySide6.QtWidgets import QMessageBox, QStyle, QApplication, QLabel, QHBoxLayout, QFrame
from PySide6.QtCore import QTimer, Qt


class _FeatureResultEmitter(QObject):
    finished = Signal(str, float, object, list)  # path, mtime, vector, keys


class _FeatureTask(QRunnable):
    def __init__(self, path, mtime, emitter):
        super().__init__()
        self.path = path
        self.mtime = mtime
        self.emitter = emitter

    def run(self):  # Executes in worker thread
        try:
            vec, keys = compute_feature_vector(self.path)
            self.emitter.finished.emit(self.path, self.mtime, vec, keys)
        except Exception:
            import logging as _logging
            _logging.exception('[FeatureAsync] Extraction failed for %s', self.path)
            self.emitter.finished.emit(self.path, self.mtime, None, [])


class LightroomMainWindow(QMainWindow):
    # Signal for thread-safe status bar updates
    status_message_signal = Signal(str)

    def __init__(self, image_paths, directory, get_sorted_images, image_info=None):
        super().__init__()
        self.directory = directory
        self.setWindowTitle("Photo Derush (Qt)")
        self.resize(1400, 800)
        # Set global font and stylesheet
        font = QFont("Segoe UI", 11)
        QApplication.instance().setFont(font)
        QApplication.instance().setStyleSheet('''
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'San Francisco', Arial, sans-serif;
                font-size: 13px;
                color: #f0f0f0;
            }
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #23272e, stop:1 #181a20);
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #23272e;
                border: none;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle {
                background: #444;
                border-radius: 5px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
            }
            QStatusBar {
                background: #23272e;
                color: #f0f0f0;
                border-top: 1px solid #333;
            }
        ''')
        self.toolbar = SettingsToolbar(self)
        self.addToolBar(self.toolbar)
        # Toast notification system
        self.toast = QLabel("")
        self.toast.setStyleSheet("background: #444; color: #fff; border-radius: 8px; padding: 8px 16px; font-size: 14px;")
        self.toast.setWindowFlags(self.toast.windowFlags() | Qt.WindowType.ToolTip)
        self.toast.hide()
        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.info_panel = InfoPanel()
        # Central stacked widget: page0 = main splitter, page1 = full image viewer (embedded)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self.splitter = QSplitter()
        self._main_page = QWidget()
        _main_layout = QVBoxLayout(self._main_page)
        _main_layout.setContentsMargins(0, 0, 0, 0)
        _main_layout.setSpacing(0)
        _main_layout.addWidget(self.splitter)
        self._stack.addWidget(self._main_page)
        self._embedded_viewer = None

        self.left_panel = QWidget()
        self.sort_by_group = False
        self.image_info = image_info or {}
        self.current_img_idx = 0
        self.sorted_images = image_paths
        self.learner = None
        self.labels_map = self._build_labels_map_from_log()
        self.logger = logging.getLogger(__name__)
        self._feature_cache = {}  # path -> (mtime, (fv, keys))
        self._feature_emitter = _FeatureResultEmitter()
        self._feature_emitter.finished.connect(self._on_async_feature_extracted)
        self._thread_pool = QThreadPool.globalInstance()
        self._pending_feature_tasks = set()
        self._last_model_save_ts = 0.0
        # Determine combined feature schema (numeric only) BEFORE using it
        try:
            self._combined_feature_names = all_feature_names(include_strings=False)
        except Exception:
            self._combined_feature_names = list(_NEW_FEATURE_NAMES)
        # Load persisted feature cache (best-effort)
        try:
            persisted_cache = load_feature_cache()
            if persisted_cache:
                self._feature_cache.update(persisted_cache)
                self.logger.info('[FeatureCache] Loaded %d cached feature vectors', len(persisted_cache))
                invalid = []
                for pth, (mt, tup) in list(self._feature_cache.items()):
                    if not isinstance(tup, tuple) or len(tup) != 2:
                        invalid.append(pth); continue
                    vec, keys = tup
                    if list(keys) == list(_NEW_FEATURE_NAMES):
                        continue
                    if list(keys) == list(self._combined_feature_names):
                        continue
                    invalid.append(pth)
                for pth in invalid:
                    self._feature_cache.pop(pth, None)
                if invalid:
                    self.logger.info('[FeatureCache] Dropped %d stale entries (schema mismatch)', len(invalid))
            else:
                self.logger.info('[FeatureCache] No persisted feature vectors loaded (empty or missing)')
        except Exception as e:
            self.logger.info('[FeatureCache] Failed loading persisted cache: %s', e)
        # Cold-start training gate (require min samples per class before training)
        self._cold_start_completed = False
        # Do not init learner if no images yet
        if self.sorted_images:
            self._init_learner()
        # create grid
        self.image_grid = ImageGrid(self.sorted_images, directory, self.info_panel, self.status, self._compute_sorted_images,
                                    image_info=self.image_info, on_open_fullscreen=self.open_fullscreen,
                                    on_select=self.on_select_image, labels_map=self.labels_map, get_feature_vector_fn=self._get_feature_vector_sync)
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
        self.toolbar.export_csv_clicked.connect(self.on_export_csv_clicked)
        self.toolbar.reset_model_clicked.connect(self.on_reset_model_clicked)
        self.toolbar.predict_sort_desc_clicked.connect(self.on_predict_sort_desc)
        self.toolbar.predict_sort_asc_clicked.connect(self.on_predict_sort_asc)
        # Backward compat original signal -> desc
        # self.toolbar.predict_sort_clicked.connect(self.on_predict_sort_desc)
        # Legacy signal no longer auto-connected to avoid double invocation causing side-effects
        self._last_sort_time = 0.0

        # After initialization, populate probabilities for all images (neutral if no learner)
        if self.sorted_images:
            self._refresh_all_keep_probs()
            first = self.get_current_image()
            if first:
                self._schedule_feature_extraction(os.path.join(self.directory, first))
                # Populate info panel (EXIF + placeholder prob) immediately
                self.refresh_keep_prob()
        # Initial status
        self._status_labels = {}
        self._init_status_widgets()
        self._update_status_bar()

        # --- Auto Unsure labeling ---
        self._auto_unsure_threshold = (0.4, 0.7)
        self._auto_unsure_assigned = set()
        self._in_sort = False
        self.sorted_images = image_paths

    def _init_status_widgets(self):
        if getattr(self, '_status_widgets_initialized', False):
            return
        try:
            from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
            container = QWidget()
            lay = QHBoxLayout(container)
            lay.setContentsMargins(4,0,4,0)
            lay.setSpacing(4)
            pill_style = ("QLabel { background:#333; color:#ddd; border-radius:10px; padding:2px 8px; font:12px 'Arial'; }"
                          "QLabel[warn='true'] { background:#6d2f00; }"
                          "QLabel[ok='true'] { background:#2e7d32; }"
                          "QLabel[bad='true'] { background:#b71c1c; }")
            container.setStyleSheet(pill_style)
            ids = [
                ('pos',''),('labels',''),('balance',''),('mode',''),('model',''),
                ('acc',''),('prec',''),('rec',''),('f1',''),('logloss',''),
                ('tasks',''),('cache',''),('progress',''),('action','')
            ]
            for i,(k,_) in enumerate(ids):
                lbl = QLabel('-')
                lbl.setObjectName(f"status_{k}")
                lay.addWidget(lbl)
                self._status_labels[k] = lbl
            lay.addStretch(1)
            self.status.addPermanentWidget(container, 1)
            self._status_widgets_initialized = True
        except Exception as e:  # noqa: PERF203
            try:
                self.status.showMessage(f"Status widgets init failed: {e}")
            except Exception:
                pass

    def _set_status_pill(self, key, text, tooltip=None, state=None):
        lbl = self._status_labels.get(key)
        if not lbl:
            return
        lbl.setText(text)
        if tooltip:
            lbl.setToolTip(tooltip)
        # reset dynamic properties
        for prop in ('ok','warn','bad'):
            try: lbl.setProperty(prop, False)
            except Exception: pass
        if state in ('ok','warn','bad'):
            try: lbl.setProperty(state, True)
            except Exception: pass
        # force style refresh
        try:
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
        except Exception:
            pass

    def _update_status_bar(self, action: str | None = None):
        if not hasattr(self, 'status') or self.status is None:
            return
        total = len(self.sorted_images)
        idx = self.current_img_idx + 1 if 0 <= self.current_img_idx < total else 0
        keep = sum(1 for v in self.labels_map.values() if v == 1)
        trash = sum(1 for v in self.labels_map.values() if v == 0)
        unsure = sum(1 for v in self.labels_map.values() if v == -1)
        labeled_def = keep + trash
        balance_pct = (keep / labeled_def * 100.0) if labeled_def else 0.0
        counts = self._definitive_label_counts()
        MIN_CLASS = 10
        if self.learner is None:
            model_state = f"model:none needT={max(0,MIN_CLASS-counts[0])} needK={max(0,MIN_CLASS-counts[1])}"
        elif not getattr(self, '_cold_start_completed', False):
            need0 = max(0, MIN_CLASS - counts[0])
            need1 = max(0, MIN_CLASS - counts[1])
            model_state = f"model:warm needT={need0} needK={need1}"
        else:
            model_state = "model:ready"
        if self.sort_by_group:
            mode = 'mode:group'
        else:
            mode = f"mode:{getattr(self, '_last_sort_mode', 'init')}"
        m = getattr(self, '_last_metrics', {}) or {}
        metrics_part = ''
        if m.get('samples'):
            # Guard against missing keys or non-finite values
            import math as _math
            def _fmt(v):
                try:
                    return f"{v:.3f}" if v is not None and _math.isfinite(float(v)) else 'nan'
                except Exception:
                    return 'nan'
            metrics_part = (f" n={m.get('samples')} acc={_fmt(m.get('acc'))} p={_fmt(m.get('prec'))} r={_fmt(m.get('rec'))} "
                             f"f1={_fmt(m.get('f1'))} pos={_fmt(m.get('pos_rate'))} ll={_fmt(m.get('logloss'))} br={_fmt(m.get('brier'))}")
        pending = len(getattr(self, '_pending_feature_tasks', []))
        feat_part = f" fp={pending}" if pending else ''
        cache_sz = len(getattr(self, '_feature_cache', {}) or {})
        cache_part = f" fc={cache_sz}" if cache_sz else ''
        # Progress detail toward cold start threshold (definitive label counts)
        progress_part = f" prog={counts[0]}/{counts[1]}" if not getattr(self, '_cold_start_completed', False) else ''
        rl_part = ''
        if hasattr(self, '_last_retrain_final_loss') and self._last_retrain_final_loss is not None:
            try:
                rl_part = f" rl={self._last_retrain_final_loss:.4f}"
            except Exception:
                rl_part = ''
        core = (f"{idx}/{total} L={labeled_def}(K{keep}/T{trash}/U{unsure}) bal={balance_pct:.1f}% {mode} "
                f"{model_state}{progress_part}{metrics_part}{feat_part}{cache_part}{rl_part}")
        if action:
            core += f" | {action}"
        try:
            self.status.showMessage(core)
        except Exception:
            pass
        # Update rich pills (do after showMessage to preserve tests)
        try:
            # use local metrics dict m (previous code referenced undefined 'metrics')
            acc_v = m.get('acc') if m else None
            prec_v = m.get('prec') if m else None
            rec_v = m.get('rec') if m else None
            f1_v = m.get('f1') if m else None
            ll_v = m.get('logloss') if m else None
            self._set_status_pill('pos', f"{idx}/{total}", "Index / Total")
            self._set_status_pill('labels', f"K{keep}/T{trash}/U{unsure}", "Keep/Trash/Unsure counts")
            self._set_status_pill('balance', f"{balance_pct:.0f}%", "Class balance (Keep %)")
            self._set_status_pill('mode', mode.replace('mode:',''), "Current sorting/group mode")
            self._set_status_pill('model', model_state, "Model state + needed samples")
            def fmt(v):
                try:
                    import math
                    return f"{float(v):.2f}" if v is not None and math.isfinite(float(v)) else '–'
                except Exception:
                    return '–'
            self._set_status_pill('acc', f"acc {fmt(acc_v)}", "Accuracy")
            self._set_status_pill('prec', f"p {fmt(prec_v)}", "Precision")
            self._set_status_pill('rec', f"r {fmt(rec_v)}", "Recall")
            self._set_status_pill('f1', f"f1 {fmt(f1_v)}", "F1 score")
            self._set_status_pill('logloss', f"ll {fmt(ll_v)}", "Log loss (lower better)")
            self._set_status_pill('tasks', f"fp {pending}" if pending else 'fp 0', "Pending feature tasks")
            self._set_status_pill('cache', f"fc {cache_sz}" if cache_sz else 'fc 0', "Feature cache size")
            prog_text = f"{counts[0]}/{counts[1]}" if not getattr(self, '_cold_start_completed', False) else 'ready'
            self._set_status_pill('progress', prog_text, "Definitive label counts (T/K) or ready")
            if action:
                self._set_status_pill('action', action, "Last action")
            else:
                self._set_status_pill('action', '', None)
        except Exception:
            pass

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
                                        on_select=self.on_select_image, labels_map=self.labels_map, get_feature_vector_fn=self._get_feature_vector_sync)
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
        # Also update probabilities for all thumbnails
        self._refresh_all_keep_probs()
        self._update_status_bar(action='images loaded')

    def update_grouping(self, image_info):
        """Update grouping metadata and refresh the grid."""
        self.logger.info("[AsyncLoad] Updating grouping metadata (%d images)", len(image_info) if image_info else 0)
        self.image_info = image_info or {}
        if self.image_grid:
            self.image_grid.image_info = self.image_info
            self.image_grid.populate_grid()
        self._update_status_bar(action='grouping updated')

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
            # Dimension migration check: compare a fresh feature vector length; if mismatch, reset
            sample_path = None
            if fv is not None:
                sample_len = len(fv)
            else:
                if self.sorted_images:
                    sample_path = os.path.join(self.directory, self.sorted_images[0])
                    fv_tuple_mig = self._get_feature_vector_sync(sample_path)
                    sample_len = len(fv_tuple_mig[0]) if fv_tuple_mig else None
                else:
                    sample_len = None
            try:
                model_feature_len = getattr(self.learner.model, 'coef_', None)
                if model_feature_len is not None:
                    model_feature_len = self.learner.model.coef_.shape[1]
                else:
                    model_feature_len = self.learner.n_features
            except Exception:
                model_feature_len = self.learner.n_features
            if sample_len is not None and sample_len != model_feature_len:
                self.logger.info('[Learner][Migration] Feature length changed persisted=%d new=%d -> reinitializing model', model_feature_len, sample_len)
                from ml.personal_learner import PersonalLearner as _PL
                self.learner = _PL(n_features=sample_len)
                # Attempt rebuild with filtered log entries of new length
                rebuild_model_from_log(self.learner, expected_n_features=sample_len)
                # Cold start may or may not be satisfied after rebuild
                self._cold_start_completed = self.learner._is_initialized
                if hasattr(self, 'image_grid'):
                    self.image_grid.learner = self.learner
            else:
                self.logger.info('[Learner] Loaded persisted model (cold start satisfied) n_features=%s', model_feature_len)
            return
        # Provided fresh fv
        if fv is not None:
            self.logger.info("[Learner] Initializing (untrained) from provided feature vector (%d dims)", len(fv))
            self.learner = PersonalLearner(n_features=len(fv))
            if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                self.logger.info("[Learner] Cold-start threshold met (%d/%d); rebuilding from log", counts[0], counts[1])
                rebuild_model_from_log(self.learner, expected_n_features=self.learner.n_features)
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
                    rebuild_model_from_log(self.learner, expected_n_features=self.learner.n_features)
                    self._cold_start_completed = True
                    if hasattr(self, 'image_grid'):
                        self.image_grid.learner = self.learner
                    return
        # Derive from first image if possible (untrained)
        if self.sorted_images:
            first_path = os.path.join(self.directory, self.sorted_images[0])
            fv_tuple = self._get_feature_vector_sync(first_path)
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

    def _on_async_feature_extracted(self, path, mtime, vec, keys):
        self._pending_feature_tasks.discard(path)
        if vec is None:
            return
        self._feature_cache[path] = (mtime, (vec, keys))
        try:
            persist_feature_cache_entry(path, mtime, vec, keys)
        except Exception as e:  # noqa: PERF203
            self.logger.info('[FeatureCache] Persist entry failed (async) for %s: %s', path, e)
        # If current image, refresh panel probability
        if self.get_current_image() and os.path.basename(path) == self.get_current_image():
            self.refresh_keep_prob()
        # Update grid probability incrementally
        if self.image_grid and self.learner is not None:
            try:
                prob = float(self.learner.predict_keep_prob([vec])[0])
                self.image_grid.update_keep_probabilities({os.path.basename(path): prob})
            except Exception as e:  # noqa: PERF203
                self.logger.info('[FeatureAsync] Prob update failed for %s: %s', path, e)

    def _get_feature_vector_sync(self, img_path):
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            return None
        cached = self._feature_cache.get(img_path)
        if cached and cached[0] == mtime:
            vec_cached, keys_cached = cached[1]
            cached_len = len(vec_cached) if hasattr(vec_cached, '__len__') else None
            combined_len = len(self._combined_feature_names)
            # Legacy short vectors (e.g., tests seeding minimal arrays) are accepted; only purge if incompatible larger.
            if cached_len is not None and cached_len > combined_len:
                self.logger.info('[FeatureCache] Purging cached vector (len=%d > combined len=%d) path=%s', cached_len, combined_len, img_path)
                self._feature_cache.pop(img_path, None)
            else:
                self.logger.debug('[FeatureCache] HIT sync (legacy acceptable) path=%s len=%s expected_max=%s', img_path, cached_len, combined_len)
                return cached[1]
        vec, keys = compute_feature_vector(img_path)
        self._feature_cache[img_path] = (mtime, (vec, keys))
        try:
            persist_feature_cache_entry(img_path, mtime, vec, keys)
        except Exception as e:  # noqa: PERF203
            self.logger.info('[FeatureCache] Persist entry failed for %s: %s', img_path, e)
        return (vec, keys)

    def _get_feature_vector(self, img_path, require_sync=False):
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            return None
        cached = self._feature_cache.get(img_path)
        if cached and cached[0] == mtime:
            vec_cached, keys_cached = cached[1]
            cached_len = len(vec_cached) if hasattr(vec_cached, '__len__') else None
            combined_len = len(self._combined_feature_names)
            if cached_len is not None and cached_len > combined_len:
                self.logger.info('[FeatureCache] Purging cached vector (len=%d > combined len=%d, async) path=%s', cached_len, combined_len, img_path)
                self._feature_cache.pop(img_path, None)
            else:
                self.logger.debug('[FeatureCache] HIT async (legacy acceptable) path=%s len=%s expected_max=%s', img_path, cached_len, combined_len)
                return cached[1]
        if require_sync:
            return self._get_feature_vector_sync(img_path)
        self._schedule_feature_extraction(img_path)
        return None

    def _schedule_feature_extraction(self, img_path):
        # Disable async extraction during pytest to reduce flakiness / segfault risk
        import sys as _sys
        if os.environ.get('PYTEST_CURRENT_TEST') or 'pytest' in _sys.modules:
            return False
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            return False
        cached = self._feature_cache.get(img_path)
        if cached and cached[0] == mtime:
            return False
        if img_path in self._pending_feature_tasks:
            return False
        self._pending_feature_tasks.add(img_path)
        task = _FeatureTask(img_path, mtime, self._feature_emitter)
        self.logger.info('[FeatureAsync] Scheduling extraction path=%s', img_path)
        self._thread_pool.start(task)
        return True

    # --------------------------------------------------------------------
    def _debounced_save_model(self, force=False, min_interval=0.5):
        now = time.time()
        if force or (now - self._last_model_save_ts) >= min_interval:
            save_model(self.learner)
            self._last_model_save_ts = now

    def _label_current_image(self, label):
        img_name = self.get_current_image()
        if img_name is None:
            return
        img_path = os.path.join(self.directory, img_name)
        fv_tuple = self._get_feature_vector_sync(img_path)
        if fv_tuple is None:
            import logging as _logging
            _logging.warning("[Learner] Feature extraction failed for %s; skipping label", img_path)
            return
        fv, _ = fv_tuple
        self.logger.info("[Label] User set label=%s for image=%s", label, img_name)
        # On-the-fly migration: if existing learner has different n_features than new vector, migrate now
        if self.learner is not None and len(fv) != getattr(self.learner, 'n_features', len(fv)):
            old_n = getattr(self.learner, 'n_features', -1)
            new_n = len(fv)
            self.logger.info('[Learner][Migration-OnLabel] Detected feature length change %d -> %d; reinitializing model', old_n, new_n)
            self.learner = PersonalLearner(n_features=new_n)
            # Rebuild from log filtering by new length
            rebuild_model_from_log(self.learner, expected_n_features=new_n)
            if hasattr(self.image_grid, 'learner'):
                self.image_grid.learner = self.learner
            # After migration, cold start status depends on initialization state
            self._cold_start_completed = self.learner._is_initialized
        if hasattr(fv, 'tolist'):
            feats_list = fv.tolist()
        else:
            # Already a list/iterable; materialize as list for JSON safety
            try:
                feats_list = list(fv)
            except Exception:
                feats_list = []
        event = {'image': img_name, 'path': img_path, 'features': feats_list, 'label': label}
        append_event(event)
        self.labels_map[img_name] = label
        # Ensure learner only for definitive labels
        if label in (0, 1):
            self._ensure_learner(fv)
            if self.learner is not None:
                counts = self._definitive_label_counts()
                MIN_CLASS_SAMPLES = 10
                # Mark cold start completed when threshold reached (for status only)
                if not self._cold_start_completed and counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                    self._cold_start_completed = True
                # Always attempt full retrain once both classes have at least one sample
                if counts[0] > 0 and counts[1] > 0:
                    from ml.persistence import load_latest_labeled_samples as _load_all, upgrade_event_log_to_current_schema as _upgrade
                    try:
                        upgraded = _upgrade()
                        if upgraded:
                            self.logger.info('[Upgrade] Event log upgraded to combined feature schema (cv+exif)')
                    except Exception as e:  # noqa: PERF203
                        self.logger.info('[Upgrade] Schema upgrade skipped: %s', e)
                    X_all, y_all, _imgs = _load_all()
                    if X_all:
                        lengths = {len(x) for x in X_all if isinstance(x, list)}
                        if lengths:
                            target_len = max(lengths)  # prefer expanded schema
                            if self.learner.n_features != target_len:
                                self.logger.info('[Learner][Migration-Retrain] Reinitializing learner %d -> %d', self.learner.n_features, target_len)
                                self.learner = PersonalLearner(n_features=target_len)
                                if hasattr(self.image_grid, 'learner'):
                                    self.image_grid.learner = self.learner
                        else:
                            target_len = self.learner.n_features
                        Xf = [x for x in X_all if isinstance(x, list) and len(x)==target_len]
                        yf = [yy for x, yy in zip(X_all, y_all) if isinstance(x, list) and len(x)==target_len]
                        if Xf:
                            self.logger.info('[Training] Full retrain (samples=%d n_features=%d countsT=%d countsK=%d)', len(Xf), target_len, counts[0], counts[1])
                            importances = self.learner.train_and_explain(Xf, yf)
                            if importances:
                                formatted = "\n".join(f"    {name}: {imp:.4f}" for name, imp in importances)
                                self.logger.info('[Training] Feature importances (sorted):\n%s', formatted)
                            else:
                                self.logger.info('[Training] Feature importances: <none>')
                            self._debounced_save_model()
                            self.logger.info('[Training] Model saved after full retrain')
                            self._evaluate_model()
                        else:
                            self.logger.info('[Training] No samples matched target feature length=%d; skipping retrain', target_len)
                else:
                    self.logger.info('[Training] Waiting for both classes before retrain (T=%d K=%d)', counts[0], counts[1])

        # update UI badge
        if hasattr(self, 'image_grid'):
            self.image_grid.update_label(img_name, label)
        if self.learner is not None:
            # Single image refresh for info panel
            self.refresh_keep_prob()
            # Batch refresh all visible thumbs (always refresh after any label change)
            self._refresh_all_keep_probs()
            logging.info("[Label] Updated keep probabilities for all images in grid after labeling %s", img_name)
        self._update_status_bar(action=f"labeled {img_name}={label}")

    def _evaluate_model(self):
        """Evaluate current learner on latest definitive labeled events (0/1) only.
        Returns a dict of metrics or None."""
        if self.learner is None:
            return None
        X, y, _images = load_latest_labeled_samples()
        if not X:
            return None
        # Filter samples whose feature length mismatches current learner (migration safety)
        expected = getattr(self.learner, 'n_features', None)
        if expected is not None:
            filtered_X = []
            filtered_y = []
            mismatched = 0
            for x_i, y_i in zip(X, y):
                try:
                    if isinstance(x_i, (list, tuple)) and len(x_i) == expected:
                        filtered_X.append(x_i)
                        filtered_y.append(y_i)
                    else:
                        mismatched += 1
                except Exception:
                    mismatched += 1
            if mismatched:
                self.logger.info('[Eval] Skipped %d samples (feature length mismatch) expected=%s', mismatched, expected)
            X, y = filtered_X, filtered_y
        if not X:
            self.logger.info('[Eval] No compatible samples after filtering; skipping evaluation')
            return None
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
            metrics = {
                'samples': len(y_np),
                'pos_rate': pos_rate,
                'acc': acc,
                'prec': prec,
                'rec': rec,
                'f1': f1,
                'logloss': ll if _np.isfinite(ll) else float('nan'),
                'brier': brier if _np.isfinite(brier) else float('nan')
            }
            self._last_metrics = metrics
            self.logger.info("[Eval][Store] %s", metrics)
            return metrics
        except Exception as e:
            self.logger.warning("[Eval] Failed evaluation: %s", e)
            return None

    def _refresh_all_keep_probs(self):
        if not self.image_grid:
            return
        names = []
        vectors = []
        # Only need probabilities for images currently in grid (respect MAX_IMAGES)
        for img_name in self.image_grid.image_paths[:self.image_grid.MAX_IMAGES]:
            img_path = os.path.join(self.directory, img_name)
            fv_tuple = self._get_feature_vector(img_path)
            if fv_tuple is None:
                logging.warning("[Predict] Feature extraction failed for %s; skipping", img_path)
                continue
            fv, _ = fv_tuple
            names.append(img_name)
            vectors.append(fv)
        if not names:
            return
        if self.learner is None:
            # Assign neutral probability
            prob_map = {n: 0.5 for n in names}
            self.image_grid.update_keep_probabilities(prob_map)
            logging.info("[Predict] Assigned neutral probabilities to %d images (no learner)", len(prob_map))
            return
        try:
            logging.info("[Predict] Predicting keep probabilities for %d images", len(vectors))
            probs = self.learner.predict_keep_prob(vectors)
            # Auto mark unsure mid-confidence
            self._auto_mark_unsure_if_needed(names, probs)
            logging.info("[Predict] Computed keep probabilities for %d images", len(probs))
            prob_map = {n: float(p) for n, p in zip(names, probs)}
            self.image_grid.update_keep_probabilities(prob_map)
            logging.info("Done updating proba in grid for %d images", len(prob_map))
            # After updating probabilities, evaluate model and store metrics; status updated centrally
            metrics = self._evaluate_model()
            if metrics:
                self._update_status_bar(action="probas updated")
            else:
                self._update_status_bar(action="probas updated (no metrics)")
        except Exception as e:
            self.logger.warning("[Predict] Failed batch probability refresh: %s", e)
            self._update_status_bar(action="proba refresh failed")

    def refresh_keep_prob(self):
        img_name = self.get_current_image()
        if img_name is None:
            return
        self._ensure_learner()  # May initialize (untrained) or rebuild from log
        img_path = os.path.join(self.directory, img_name)
        keep_prob = None
        explanations = []
        if self.learner is None:
            self.logger.info("[Predict] Skipping keep_prob prediction (no learner)")
        else:
            fv_tuple = self._get_feature_vector_sync(img_path)
            if fv_tuple is None:
                self.logger.warning("[Predict] Feature extraction failed for %s; cannot predict", img_path)
            else:
                fv, keys = fv_tuple
                try:
                    self.logger.info("[Predict] Computing keep probability for image=%s", img_name)
                    keep_prob = float(self.learner.predict_keep_prob([fv])[0]) if fv is not None else None
                    if keep_prob is not None:
                        self.logger.info("[Predict] keep_prob=%.4f image=%s", keep_prob, img_name)
                        # Auto mark unsure for single image path
                        self._auto_mark_unsure_if_needed([img_name], [keep_prob])
                        explanations = self._explain_vector(fv, keys)
                except Exception as e:  # noqa: PERF203
                    self.logger.warning("[Predict] Prediction failed for %s: %s", img_path, e)
        # Always update info panel (ensures EXIF is displayed even before model exists)
        self.info_panel.update_info(img_name, img_path, "-", "-", "-", keep_prob=keep_prob, explanations=explanations)
        if keep_prob is not None:
            logging.info("[Predict] Updated keep_prob for image=%s: %.4f", img_name, keep_prob)

    def on_export_csv_clicked(self):
        # Enhanced export with EXIF subset + explanation tokens
        subset_exif_keys = ["DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
        rows = []
        from .utils import extract_exif
        for img_name in self.sorted_images:
            path = os.path.join(self.directory, img_name)
            label = self.labels_map.get(img_name, '')
            prob = ''
            explanations = ''
            if self.learner is not None:
                fv_tuple = self._get_feature_vector_sync(path)
                if fv_tuple is not None:
                    fv, keys = fv_tuple
                    try:
                        p = float(self.learner.predict_keep_prob([fv])[0])
                        prob = f"{p:.4f}"
                        expl_list = self._explain_vector(fv, keys)
                        explanations = ';'.join(f"{name}:{'+' if good else '-'}" for name, good in expl_list)
                    except Exception:
                        pass
            exif = {}
            try:
                exif = extract_exif(path) or {}
            except Exception:
                pass
            exif_vals = [str(exif.get(k, '')) for k in subset_exif_keys]
            rows.append([path, label, prob, explanations] + exif_vals)
        header = ['path', 'label', 'keep_prob', 'explanations'] + subset_exif_keys
        try:
            with open('labels.csv', 'w') as f:
                f.write(','.join(header) + '\n')
                for row in rows:
                    f.write(','.join(str(x) for x in row) + '\n')
            self.logger.info('[Export] Wrote labels.csv with %d rows (enhanced)', len(rows))
        except Exception as e:
            self.logger.warning('[Export] Failed writing labels.csv: %s', e)

    def set_status_message(self, message: str):
        """
        Public method to update the status bar with a custom message.
        Can be called from any thread (uses Qt signal for thread safety).
        Intended for async progress reporting (e.g., hashing, loading, etc).
        """
        self.status_message_signal.emit(message)

    def show_toast(self, message, duration=2000):
        self.toast.setText(message)
        self.toast.adjustSize()
        self.toast.move(self.geometry().center() - self.toast.rect().center())
        self.toast.show()
        QTimer.singleShot(duration, self.toast.hide)

    # --- Keyboard shortcuts in grid / main window mode ---
    def keyPressEvent(self, e):  # noqa: N802
        from PySide6.QtCore import Qt
        key = e.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A, Qt.Key.Key_H):
            # Prev image
            if self.current_img_idx > 0:
                self.current_img_idx -= 1
                self.refresh_keep_prob()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_L):
            if self.current_img_idx < len(self.sorted_images) - 1:
                self.current_img_idx += 1
                self.refresh_keep_prob()
        elif key in (Qt.Key.Key_Space,):
            # toggle keep / trash
            img = self.get_current_image()
            cur = self.labels_map.get(img)
            if cur == 1:
                self._label_current_image(0)
            else:
                self._label_current_image(1)
        else:
            return super().keyPressEvent(e)

    # --- Explanation helpers ---
    def _explain_vector(self, fv, keys=None, top_n=5):
        """Return list of (token, good_bool) for the given feature vector using linear model contributions.
        Falls back to empty list if model or coefficients unavailable."""
        if self.learner is None:
            return []
        model = getattr(self.learner, 'model', None)
        coef = getattr(model, 'coef_', None)
        if coef is None:
            return []
        try:
            import numpy as _np
            w = coef[0] if hasattr(coef, '__len__') and len(coef.shape) == 2 else coef
            fv_arr = _np.asarray(fv, dtype=float)
            if fv_arr.shape[0] != w.shape[-1]:
                return []
            contrib = w * fv_arr
            # choose feature names order
            names = None
            if keys and len(keys) == len(contrib):
                names = list(keys)
            elif getattr(self.learner, 'feature_names', None) and len(self.learner.feature_names) == len(contrib):
                names = list(self.learner.feature_names)
            else:
                names = [f'f{i}' for i in range(len(contrib))]
            idxs = list(range(len(contrib)))
            idxs.sort(key=lambda i: abs(contrib[i]), reverse=True)
            out = []
            for i in idxs[:top_n]:
                val = contrib[i]
                label = names[i]
                good = val >= 0
                # simple prettify / collapse overly long names
                if len(label) > 22:
                    label = label[:20] + '…'
                out.append((label, good))
            return out
        except Exception:
            return []

    def _auto_mark_unsure_if_needed(self, names, probs):
        """Auto-mark unlabeled mid-confidence images as Unsure (-1).
        Avoid duplicate event log entries using _auto_unsure_assigned set."""
        low, high = self._auto_unsure_threshold
        for n, p in zip(names, probs):
            if n in self.labels_map:  # already user labeled or previously auto-labeled
                continue
            if n in self._auto_unsure_assigned:
                continue
            if low <= p <= high:
                # attempt feature vector for log; skip persistence on failure
                path = os.path.join(self.directory, n)
                fv_tuple = self._get_feature_vector_sync(path)
                feats_list = []
                if fv_tuple:
                    fv, _k = fv_tuple
                    try:
                        feats_list = fv.tolist() if hasattr(fv, 'tolist') else list(fv)
                    except Exception:
                        feats_list = []
                event = {'image': n, 'path': path, 'features': feats_list, 'label': -1, 'auto':'unsure'}
                try:
                    append_event(event)
                except Exception:
                    pass
                self.labels_map[n] = -1
                self._auto_unsure_assigned.add(n)
                if self.image_grid:
                    self.image_grid.update_label(n, -1)
                self.logger.info('[AutoUnsure] Marked %s prob=%.3f', n, p)

    def on_sort_by_group_toggled(self, checked: bool):
        self.sort_by_group = bool(checked)
        if self.image_grid:
            self.image_grid.populate_grid()
        self._update_status_bar(action='group sort on' if checked else 'group sort off')

    def on_select_image(self, idx: int):
        if 0 <= idx < len(self.sorted_images):
            self.current_img_idx = idx
            self.refresh_keep_prob()
            self._update_status_bar(action=f'select {idx+1}')

    def open_fullscreen(self, start_index: int, path: str):
        if getattr(self, '_in_sort', False):
            return
        # Defensive bounds
        if not self.sorted_images:
            return
        if not (0 <= start_index < len(self.sorted_images)):
            start_index = 0
        self.current_img_idx = start_index
        seq_paths = [os.path.join(self.directory, n) for n in self.sorted_images]
        from .viewer import open_full_image_qt
        open_full_image_qt(path, on_keep=self.on_keep_clicked, on_trash=self.on_trash_clicked,
                           on_unsure=self.on_unsure_clicked, image_sequence=seq_paths,
                           start_index=start_index, on_index_change=self._on_viewer_index_change,
                           parent_main_window=self)

    def _on_viewer_index_change(self, path: str, idx: int):
        # Sync main window index when embedded viewer navigates
        if 0 <= idx < len(self.sorted_images):
            self.current_img_idx = idx
            self.refresh_keep_prob()
            self._update_status_bar(action=f'viewer {idx+1}')

    def _show_embedded_viewer(self, viewer_widget: QWidget):
        # Add viewer as second page in stack if not already
        if viewer_widget is None:
            return
        # Remove previous viewer page if exists
        if getattr(self, '_embedded_viewer', None) is not None:
            try:
                old = self._embedded_viewer
                self._stack.removeWidget(old)
                old.setParent(None)
            except Exception:
                pass
        self._embedded_viewer = viewer_widget
        self._stack.addWidget(viewer_widget)
        self._stack.setCurrentWidget(viewer_widget)
        self._update_status_bar(action='viewer open')

    def _restore_from_viewer(self):
        # Return to main splitter page
        self._stack.setCurrentWidget(self._main_page)
        if getattr(self, '_embedded_viewer', None) is not None:
            try:
                vw = self._embedded_viewer
                self._stack.removeWidget(vw)
                vw.setParent(None)
            except Exception:
                pass
            self._embedded_viewer = None
        self._update_status_bar(action='viewer closed')

    def _collect_probabilities_for_sort(self):
        names = list(self.sorted_images)
        if not names:
            return {}
        if self.learner is None:
            return {n: 0.5 for n in names}
        vecs = []
        valid_names = []
        for n in names:
            path = os.path.join(self.directory, n)
            cached = self._feature_cache.get(path)
            if cached:
                vecs.append(cached[1][0])
                valid_names.append(n)
        if not vecs:
            return {n:0.5 for n in names}
        try:
            probs = self.learner.predict_keep_prob(vecs)
        except Exception:
            # Fallback: if learner returns binary probs shape (N,2)
            try:
                import numpy as _np
                probs = _np.asarray(probs)
                if probs.ndim == 2 and probs.shape[1] == 2:
                    probs = probs[:,1]
            except Exception:
                return {n:0.5 for n in names}
        # If predict_keep_prob returns 1D array -> fine
        import numpy as _np
        probs_arr = _np.asarray(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            probs_arr = probs_arr[:,1]
        prob_map = {n: float(p) for n,p in zip(valid_names, probs_arr)}
        # Assign neutral to missing
        for n in names:
            if n not in prob_map:
                prob_map[n] = 0.5
        return prob_map

    def on_predict_sort_desc(self):
        self._in_sort = True
        prob_map = self._collect_probabilities_for_sort()
        self.sorted_images = sorted(self.sorted_images, key=lambda n: prob_map.get(n,0.5), reverse=True)
        if self.image_grid:
            self.image_grid.image_paths = self.sorted_images
            self.image_grid.populate_grid()
        self._last_sort_mode = 'desc'
        self.current_img_idx = 0 if self.sorted_images else -1
        self._update_status_bar(action='sort desc')
        self._in_sort = False

    def on_predict_sort_asc(self):
        self._in_sort = True
        prob_map = self._collect_probabilities_for_sort()
        self.sorted_images = sorted(self.sorted_images, key=lambda n: prob_map.get(n,0.5))
        if self.image_grid:
            self.image_grid.image_paths = self.sorted_images
            self.image_grid.populate_grid()
        self._last_sort_mode = 'asc'
        self.current_img_idx = 0 if self.sorted_images else -1
        self._update_status_bar(action='sort asc')
        self._in_sort = False

    def on_reset_model_clicked(self):
        try:
            clear_model_and_log()
            self.logger.info('[Model] Cleared personal model + event log')
        except Exception as e:  # noqa: PERF203
            self.logger.warning('[Model] Failed clearing model/log: %s', e)
        self.learner = None
        if hasattr(self.image_grid, 'learner'):
            self.image_grid.learner = None
        # Rebuild labels map (will be empty except maybe unsure / previous entries ignored)
        self.labels_map = self._build_labels_map_from_log()
        # Reset probabilities to neutral
        if self.image_grid:
            neutral = {n: 0.5 for n in self.image_grid.image_paths[:self.image_grid.MAX_IMAGES]}
            self.image_grid.update_keep_probabilities(neutral)
        self._cold_start_completed = False
        self._update_status_bar(action='model reset')

    def toggle_dark_mode(self):
        if self.is_dark_mode:
            QApplication.instance().setStyleSheet('''
                QWidget {
                    font-family: 'Segoe UI', 'Roboto', 'San Francisco', Arial, sans-serif;
                    font-size: 13px;
                    color: #23272e;
                }
                QMainWindow {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f0f0f0, stop:1 #e0e0e0);
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    background: #e0e0e0;
                    border: none;
                    width: 10px;
                    margin: 0px;
                }
                QScrollBar::handle {
                    background: #bbb;
                    border-radius: 5px;
                }
                QScrollBar::add-line, QScrollBar::sub-line {
                    background: none;
                }
                QStatusBar {
                    background: #e0e0e0;
                    color: #23272e;
                    border-top: 1px solid #bbb;
                }
            ''')
            self.is_dark_mode = False
        else:
            QApplication.instance().setStyleSheet('''
                QWidget {
                    font-family: 'Segoe UI', 'Roboto', 'San Francisco', Arial, sans-serif;
                    font-size: 13px;
                    color: #f0f0f0;
                }
                QMainWindow {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #23272e, stop:1 #181a20);
                }
                QScrollBar:vertical, QScrollBar:horizontal {
                    background: #23272e;
                    border: none;
                    width: 10px;
                    margin: 0px;
                }
                QScrollBar::handle {
                    background: #444;
                    border-radius: 5px;
                }
                QScrollBar::add-line, QScrollBar::sub-line {
                    background: none;
                }
                QStatusBar {
                    background: #23272e;
                    color: #f0f0f0;
                    border-top: 1px solid #333;
                }
            ''')
            self.is_dark_mode = True
