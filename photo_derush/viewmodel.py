from PySide6.QtCore import QObject, Signal
import os
import logging
from ml.features import feature_vector, all_feature_names
from ml.personal_learner import PersonalLearner
from ml.persistence import (
    save_model, append_event, rebuild_model_from_log, clear_model_and_log,
    load_model, load_feature_cache, persist_feature_cache_entry,
    latest_labeled_events, load_latest_labeled_samples
)
from ml.features_cv import FEATURE_NAMES as _NEW_FEATURE_NAMES

class LightroomViewModel(QObject):
    images_changed = Signal(list)
    labels_changed = Signal(dict)
    keep_probs_changed = Signal(dict)
    status_changed = Signal(str)
    model_state_changed = Signal(object)

    def __init__(self, image_paths, directory, image_info=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.directory = directory
        self.image_info = image_info or {}
        self.sorted_images = image_paths
        self.labels_map = self._build_labels_map_from_log()
        self.learner = None
        self._feature_cache = {}
        self._combined_feature_names = self._get_combined_feature_names()
        self._cold_start_completed = False
        self._init_feature_cache()
        if self.sorted_images:
            self._init_learner()

    def _get_combined_feature_names(self):
        try:
            return all_feature_names(include_strings=False)
        except Exception:
            return list(_NEW_FEATURE_NAMES)

    def _init_feature_cache(self):
        try:
            persisted_cache = load_feature_cache()
            if persisted_cache:
                self._feature_cache.update(persisted_cache)
        except Exception as e:
            self.logger.info('[FeatureCache] Failed loading persisted cache: %s', e)

    def _build_labels_map_from_log(self):
        labels = {}
        latest = latest_labeled_events()
        for img, ev in latest.items():
            lbl = ev.get('label')
            if lbl is not None:
                labels[os.path.basename(img)] = lbl
        return labels

    def _init_learner(self):
        if self.sorted_images:
            self._ensure_learner()

    def _ensure_learner(self, fv=None):
        if self.learner is not None:
            return
        MIN_CLASS_SAMPLES = 10
        counts = self._definitive_label_counts()
        try:
            persisted = load_model()
        except Exception as e:
            persisted = None
            self.logger.warning("[Learner] Failed to load persisted model: %s", e)
        if persisted is not None:
            self.learner = persisted
            self._cold_start_completed = True
            return
        if fv is not None:
            self.learner = PersonalLearner(n_features=len(fv))
            if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                rebuild_model_from_log(self.learner, expected_n_features=self.learner.n_features)
                self._cold_start_completed = True
            return
        if counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
            latest = latest_labeled_events()
            for ev in latest.values():
                feats = ev.get('features')
                if feats:
                    self.learner = PersonalLearner(n_features=len(feats))
                    rebuild_model_from_log(self.learner, expected_n_features=self.learner.n_features)
                    self._cold_start_completed = True
                    return
        if self.sorted_images:
            first_path = os.path.join(self.directory, self.sorted_images[0])
            fv_tuple = self._get_feature_vector_sync(first_path)
            if fv_tuple is not None:
                fv0, _ = fv_tuple
                self.learner = PersonalLearner(n_features=len(fv0))

    def _definitive_label_counts(self):
        counts = {0: 0, 1: 0}
        latest = latest_labeled_events()
        for ev in latest.values():
            lbl = ev.get('label')
            if lbl in (0,1):
                counts[lbl] += 1
        return counts[0], counts[1]

    def _get_feature_vector_sync(self, img_path):
        try:
            mtime = os.path.getmtime(img_path)
        except OSError:
            return None
        cached = self._feature_cache.get(img_path)
        if cached and cached[0] == mtime:
            return cached[1]
        vec, keys = feature_vector(img_path)
        self._feature_cache[img_path] = (mtime, (vec, keys))
        try:
            persist_feature_cache_entry(img_path, mtime, vec, keys)
        except Exception as e:
            self.logger.info('[FeatureCache] Persist entry failed for %s: %s', img_path, e)
        return (vec, keys)

    def label_image(self, img_name, label):
        if not img_name:
            self.logger.info("[ViewModel] No image to label.")
            return
        img_path = os.path.join(self.directory, img_name)
        fv_tuple = self._get_feature_vector_sync(img_path)
        if fv_tuple is None:
            self.logger.warning("[ViewModel] Feature extraction failed for %s; skipping label", img_path)
            return
        fv, _ = fv_tuple
        if self.learner is not None and len(fv) != getattr(self.learner, 'n_features', len(fv)):
            old_n = getattr(self.learner, 'n_features', -1)
            new_n = len(fv)
            self.logger.info('[ViewModel][Migration-OnLabel] Feature length change %d -> %d; reinitializing model', old_n, new_n)
            self.learner = PersonalLearner(n_features=new_n)
            rebuild_model_from_log(self.learner, expected_n_features=new_n)
            self._cold_start_completed = self.learner._is_initialized
            self.model_state_changed.emit(self.learner)
        feats_list = fv.tolist() if hasattr(fv, 'tolist') else list(fv)
        event = {'image': img_name, 'path': img_path, 'features': feats_list, 'label': label}
        append_event(event)
        self.labels_map[img_name] = label
        self.labels_changed.emit(self.labels_map)
        if label in (0, 1):
            self._ensure_learner(fv)
            if self.learner is not None:
                counts = self._definitive_label_counts()
                MIN_CLASS_SAMPLES = 10
                if not self._cold_start_completed and counts[0] >= MIN_CLASS_SAMPLES and counts[1] >= MIN_CLASS_SAMPLES:
                    self._cold_start_completed = True
                if counts[0] > 0 and counts[1] > 0:
                    from ml.persistence import load_latest_labeled_samples as _load_all, upgrade_event_log_to_current_schema as _upgrade
                    try:
                        upgraded = _upgrade()
                        if upgraded:
                            self.logger.info('[Upgrade] Event log upgraded to combined feature schema (cv+exif)')
                    except Exception as e:
                        self.logger.info('[Upgrade] Schema upgrade skipped: %s', e)
                    X_all, y_all, _imgs = _load_all()
                    if X_all:
                        lengths = {len(x) for x in X_all if isinstance(x, list)}
                        if lengths:
                            target_len = max(lengths)
                            if self.learner.n_features != target_len:
                                self.logger.info('[ViewModel][Migration-Retrain] Reinitializing learner %d -> %d', self.learner.n_features, target_len)
                                self.learner = PersonalLearner(n_features=target_len)
                                self.model_state_changed.emit(self.learner)
                        else:
                            target_len = self.learner.n_features
                        Xf = [x for x in X_all if isinstance(x, list) and len(x)==target_len]
                        yf = [yy for x, yy in zip(X_all, y_all) if isinstance(x, list) and len(x)==target_len]
                        if Xf:
                            self.logger.info('[ViewModel][Training] Full retrain (samples=%d n_features=%d countsT=%d countsK=%d)', len(Xf), target_len, counts[0], counts[1])
                            self.learner.train_and_explain(Xf, yf)
                            save_model(self.learner)
                            self.logger.info('[ViewModel][Training] Model saved after full retrain')
                            self.model_state_changed.emit(self.learner)
                            self.logger.info('[ViewModel] Emitting status_changed: Training complete')
                            self.status_changed.emit("Training complete")
                            return  # Prevents emitting 'Labeled ...' below if training occurred
        self.status_changed.emit(f"Labeled {img_name}={label}")
        # Optionally, emit keep_probs_changed if probabilities should be refreshed

    def get_sorted_images(self, sort_by_group=False):
        if sort_by_group and self.image_info:
            def group_key(img):
                info = self.image_info.get(img, {})
                group = info.get("group")
                return (group if group is not None else 999999, img)
            return sorted(self.sorted_images, key=group_key)
        return self.sorted_images
