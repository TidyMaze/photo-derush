"""AutoLabelManager: encapsulates auto-labeling, prediction, and retraining logic.

This extracts responsibilities from the monolithic viewmodel for SRP:
  - Manage thresholds / settings
  - Schedule and run retraining tasks
  - Batch predictions and probability bookkeeping
  - Apply / refresh auto labels

Can be used independently of UI via BrainInterface, enabling headless use or plugins.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .brain_interface import BrainCallbacks, ImageDataProvider, TaskRunner

try:  # lightweight optional import (settings access)
    from .settings import get_setting
except Exception:  # pragma: no cover

    def get_setting(key: str, default=None):  # type: ignore
        return default


class AutoLabelManager:
    def __init__(
        self,
        data_provider: ImageDataProvider,
        callbacks: BrainCallbacks | None = None,
        task_runner: TaskRunner | None = None,
        images: list[str] | None = None,
    ):
        """Initialize AutoLabelManager with data provider and optional callbacks.

        Args:
            data_provider: Provides access to image data (ImageModel or adapter)
            callbacks: Optional callbacks for UI integration (can be None for headless)
            task_runner: Optional task runner for background execution (can be None for sync)
            images: Optional list of image filenames (for compatibility with viewmodel)
        """
        self.data = data_provider
        from .brain_interface import BrainCallbacks

        self.callbacks = callbacks or BrainCallbacks()
        self.task_runner = task_runner
        self.images = images or []
        # Backward compatibility: if PhotoViewModel passed, extract components
        if hasattr(data_provider, "model") and hasattr(data_provider, "label_changed"):
            # Legacy PhotoViewModel passed - wrap it
            from .brain_interface import BrainAdapter

            self.data = BrainAdapter(data_provider.model)
            if callbacks is None:
                # Create callbacks from PhotoViewModel signals
                def on_label_changed(fname: str, state: str | None):
                    try:
                        data_provider.label_changed.emit(fname, state or "")
                    except Exception:
                        pass

                def on_prediction_updated(fname: str, prob: float, label: str):
                    try:
                        if hasattr(data_provider, "prediction_updated"):
                            data_provider.prediction_updated.emit(fname, prob, label)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                def on_state_snapshot():
                    try:
                        if hasattr(data_provider, "_emit_state_snapshot"):
                            data_provider._emit_state_snapshot()
                    except Exception:
                        pass

                def on_model_stats_changed(stats: dict):
                    try:
                        if hasattr(data_provider, "model_stats_changed"):
                            data_provider.model_stats_changed.emit(stats)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                self.callbacks = BrainCallbacks(
                    on_label_changed=on_label_changed,
                    on_prediction_updated=on_prediction_updated,
                    on_state_snapshot=on_state_snapshot,
                    on_model_stats_changed=on_model_stats_changed,
                )
            if task_runner is None and hasattr(data_provider, "_tasks"):
                self.task_runner = data_provider._tasks
            if images is None and hasattr(data_provider, "images"):
                self.images = data_provider.images
        # Config / thresholds
        self.enabled = bool(get_setting("auto_label_enabled", True))
        # Lazy import: avoid importing training_core (sklearn) during startup
        # Use default path directly, only import training module when actually needed
        _default_model_path = os.path.expanduser("~/.photo-derush-keep-trash-model.joblib")
        self.model_path = get_setting("auto_label_model_path", _default_model_path)
        # Updated default thresholds (was 0.8 / 0.2)
        self.keep_thr = float(get_setting("auto_label_keep_threshold", 0.7) or 0.7)
        self.trash_thr = float(get_setting("auto_label_trash_threshold", 0.3) or 0.3)
        # Auto-apply (very high confidence) thresholds (user requested: trash=0.2, keep=0.8)
        self.keep_auto_thr = float(get_setting("auto_label_autoapply_keep_threshold", 0.8) or 0.8)
        self.trash_auto_thr = float(get_setting("auto_label_autoapply_trash_threshold", 0.2) or 0.2)
        if self.keep_thr <= self.trash_thr:  # sanity
            self.keep_thr, self.trash_thr = 0.7, 0.3
        # Retrain state
        self._retrain_in_progress = False
        self._retrain_pending = False
        self._retrain_lock = threading.Lock()
        self._last_retrain_time = 0.0
        self._retrain_min_interval = 1.5
        # Prediction state
        self._predict_in_progress = False
        self._predict_lock = threading.Lock()
        self._predict_pending = False
        self.predicted_labels: dict[str, str] = {}
        self.predicted_probabilities: dict[str, float] = {}
        self.auto_assigned: set[str] = set()
        # NaN suppression guards
        self._nan_retrain_attempts = 0
        self._nan_retrain_attempts_max = 2
        self._nan_retrain_min_interval = 5.0
        self._last_nan_retrain_time = 0.0
        # Model stats (TrainingResult)
        self.model_stats = None

        # Initialize review manager for high-confidence auto-labels
        self.review_manager: object | None = None
        try:
            from .auto_label_review import AutoLabelReviewManager

            review_threshold = float(get_setting("auto_label_review_threshold", 0.9) or 0.9)
            review_mgr = AutoLabelReviewManager(self.data.directory, confidence_threshold=review_threshold)
            self.review_manager = review_mgr  # type: ignore[assignment]
        except Exception as e:
            logging.warning(f"[auto-label] Failed to initialize review manager: {e}")
            self.review_manager = None
        # Remove dynamic threshold caches
        # self._dynamic_keep_thr: float | None = None
        # self._dynamic_trash_thr: float | None = None
        # Remove weighted mode (no dynamic adjustments)
        self.weighted_mode = bool(get_setting("auto_label_weighted_enabled", False))
        self._dynamic_keep_thr = None
        self._dynamic_trash_thr = None
        # Load overrides from settings file immediately
        try:
            self.refresh_settings()
        except Exception as e:
            logging.debug(f"[auto-label] Initial refresh_settings failed: {e}")

        self.protected_manual_labels: set[str] = set()  # filenames of manual labels to never alter
        # Debounce timer for label change -> re-predict
        self._label_change_timer: threading.Timer | None = None
        self._label_change_lock = threading.Lock()
        self._label_change_delay = float(get_setting("auto_label_repredict_delay", 0.18) or 0.18)

        # Note: Label change notifications are handled via callbacks.on_label_changed
        # External code should call notify_label_changed() when labels change

    def notify_label_changed(self, filename: str, new_state: str | None = None):
        """Called when any label changes (external notification).

        Debounces rapid changes and triggers a full re-prediction after
        a short delay so downstream auto-labeling / refresh happens.
        """
        logging.info(f"[auto-label] notify_label_changed called for {filename} with {new_state}")
        try:
            # If this filename was auto-assigned by us, ignore to avoid feedback loops
            try:
                path = self.data.get_image_path(filename) if filename else None
                if path and path in self.auto_assigned:
                    return
            except Exception:
                pass

            with self._label_change_lock:
                # Cancel previous timer if scheduled
                try:
                    if self._label_change_timer is not None:
                        self._label_change_timer.cancel()
                except Exception:
                    pass

                # Schedule a re-predict after a short debounce delay
                try:
                    timer = threading.Timer(self._label_change_delay, self._on_label_change_timer_cb)
                    timer.daemon = True
                    timer.start()
                    self._label_change_timer = timer
                except Exception as e:
                    raise
        except Exception as e:
            logging.exception("[auto-label] _on_label_changed failed")

    def _on_label_change_timer_cb(self):
        try:
            # Skip prediction update if retraining is in progress or pending
            # The post-retrain workflow will handle predictions with the new model
            with self._retrain_lock:
                if self._retrain_in_progress or self._retrain_pending:
                    logging.debug("[auto-label] Skipping re-predict, retrain in progress")
                    return
            # Trigger a full prediction update which will in turn apply auto-labels
            self.update_predictions_async()
        except Exception as e:
            logging.exception("[auto-label] re-predict task failed")

    # ---------------- Configuration Refresh -----------------
    def refresh_settings(self):
        try:
            self.enabled = bool(get_setting("auto_label_enabled", self.enabled))

            keep = float(get_setting("auto_label_keep_threshold", self.keep_thr))
            trash = float(get_setting("auto_label_trash_threshold", self.trash_thr))

            if keep <= trash:
                logging.warning(f"Invalid thresholds: keep={keep} <= trash={trash}, using defaults")
            else:
                self.keep_thr = keep
                self.trash_thr = trash

            mp = get_setting("auto_label_model_path", self.model_path) or self.model_path
            self.model_path = mp
            # Weighted mode can be toggled in settings
            self.weighted_mode = bool(get_setting("auto_label_weighted_enabled", self.weighted_mode))
            # Auto-apply thresholds may be adjusted at runtime
            try:
                self.keep_auto_thr = float(
                    get_setting("auto_label_autoapply_keep_threshold", self.keep_auto_thr) or self.keep_auto_thr
                )
                self.trash_auto_thr = float(
                    get_setting("auto_label_autoapply_trash_threshold", self.trash_auto_thr) or self.trash_auto_thr
                )
            except Exception:
                pass
        except Exception as e:
            logging.debug(f"Failed to refresh settings: {e}")

    # ---------------- Prediction -----------------
    def compute_dynamic_thresholds(self) -> tuple[float, float]:
        if not self.weighted_mode or not self.model_stats:
            return self.keep_thr, self.trash_thr
        try:
            n_keep = getattr(self.model_stats, "n_keep", 0)
            n_trash = getattr(self.model_stats, "n_trash", 0)
            if n_keep <= 0 or n_trash <= 0:
                return self.keep_thr, self.trash_thr
            majority = "keep" if n_keep >= n_trash else "trash"
            ratio = max(n_keep, n_trash) / min(n_keep, n_trash)
            keep_eff = self.keep_thr
            trash_eff = self.trash_thr
            if ratio > 1.2:
                if majority == "keep":
                    # raise keep threshold slightly
                    keep_eff = min(1.0, keep_eff + min(0.08, (ratio - 1.0) * 0.04))
                else:
                    # lower trash threshold slightly
                    trash_eff = max(0.0, trash_eff - min(0.05, (ratio - 1.0) * 0.03))
            if keep_eff <= trash_eff:
                mid = (keep_eff + trash_eff) / 2.0
                keep_eff = min(1.0, mid + 0.05)
                trash_eff = max(0.0, mid - 0.05)
            self._dynamic_keep_thr = keep_eff
            self._dynamic_trash_thr = trash_eff
            return keep_eff, trash_eff
        except Exception:
            return self.keep_thr, self.trash_thr

    def set_weighted_mode(self, enabled: bool):
        self.weighted_mode = bool(enabled)
        # recompute for visibility
        self.compute_dynamic_thresholds()

    def classify(self, prob: float) -> str:
        if prob != prob:
            return ""
        keep_eff, trash_eff = self.compute_dynamic_thresholds()
        if prob >= keep_eff:
            return "keep"
        if prob <= trash_eff:
            return "trash"
        return ""

    def _process_probability(self, fname: str, prob: float, auto_assign: bool = False):
        """Central probability -> (predicted_probabilities, predicted_labels, optional auto assignment + signals).
        DRY helper used by all prediction paths."""
        if prob != prob:  # NaN -> skip mapping
            return
        self.predicted_probabilities[fname] = prob
        label = self.classify(prob)
        self.predicted_labels[fname] = label
        if label and self.callbacks.on_prediction_updated:
            try:
                self.callbacks.on_prediction_updated(fname, prob, label)
            except Exception:
                logging.exception("[auto-label] on_prediction_updated callback failed for %s", fname)
        # Decide whether to auto-assign the label into repository.
        # Conditions:
        #  - auto_assign flag provided OR
        #  - prediction exceeds very-high-confidence auto thresholds
        # Also require auto-labeling to be enabled and a non-empty label.
        try:
            should_auto = False
            if label and self.enabled:
                if auto_assign:
                    should_auto = True
                else:
                    # For 'keep', probability near 1.0; for 'trash', probability near 0.0
                    if label == "keep" and prob >= getattr(self, "keep_auto_thr", self.keep_auto_thr):
                        should_auto = True
                    if label == "trash" and prob <= getattr(self, "trash_auto_thr", self.trash_auto_thr):
                        should_auto = True

            if should_auto and label:
                path = self.data.get_image_path(fname)
                if path and not self.data.get_state(path):
                    if self._safe_set_auto_state(path, fname, label):
                        self.auto_assigned.add(path)
                        if self.callbacks.on_label_changed:
                            try:
                                self.callbacks.on_label_changed(fname, label)
                            except Exception:
                                logging.exception("[auto-label] on_label_changed callback failed for %s", fname)
        except Exception:
            logging.exception("[auto-label] auto-assign failed for %s", fname)

    def _process_predictions(self, filenames: list[str], probs: list[float]) -> tuple[dict, dict]:
        """Process raw probabilities into label and probability dicts.

        Also flags high-confidence predictions for human review.
        Updates self.predicted_labels and predicted_probabilities incrementally
        and emits incremental updates to the view for real-time display.
        """
        new_map = {}
        new_probs = {}
        flagged_count = 0

        for fname, prob in zip(filenames, probs):
            new_probs[fname] = 0.0 if prob != prob else prob
            self._process_probability(fname, prob, auto_assign=False)
            new_map[fname] = self.predicted_labels.get(fname, "")

            # Flag high-confidence predictions for review (only if label predicted)
            if self.review_manager and new_map[fname]:
                if self.review_manager.flag_for_review(fname, new_map[fname], prob):  # type: ignore[attr-defined]
                    flagged_count += 1

        if flagged_count > 0:
            logging.info(f"[auto-label] Flagged {flagged_count} predictions for review")

        return new_map, new_probs

    def predict_with_progress(self):
        """Predict on ALL images (not just displayed) to maintain feature cache."""
        try:
            from .training import predict_keep_probability

            if not (self.model_path and os.path.isfile(self.model_path)):
                self.predicted_labels.clear()
                self.predicted_probabilities.clear()
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
                return

            all_files = self.data.get_image_files()
            paths = [self.data.get_image_path(f) for f in all_files]
            logging.info(f"[predict] start total={len(paths)}")

            def _internal_cb(current, tot, detail):
                if tot and current % max(1, tot // 10) == 0:
                    logging.info(f"[predict] progress {current}/{tot} {detail}")

            probs = predict_keep_probability(paths, model_path=self.model_path, progress_callback=_internal_cb)
            # If ALL predictions are NaN (broken model), clear stale predictions
            nan_count = sum(1 for p in probs if p != p)
            if nan_count == len(probs) and len(probs) > 0:
                logging.warning("[predict] All predictions are NaN (broken model) - clearing stale predictions")
                # Clear predictions and prevent concurrent updates from restoring them
                with self._predict_lock:
                    self.predicted_labels.clear()
                    self.predicted_probabilities.clear()
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
                return
            new_map, new_probs = self._process_predictions(all_files, probs)

            # Already updated incrementally in _process_predictions, just emit final state
            logging.info(f"[predict] done total predictions={len(new_map)}, stored={len(self.predicted_probabilities)}")
            # Ensure state snapshot includes updated predictions
            if self.callbacks.on_state_snapshot:
                try:
                    self.callbacks.on_state_snapshot()
                except Exception:
                    pass
            logging.debug(f"[predict] State snapshot emitted with {len(self.predicted_probabilities)} probabilities")

            # If auto-labeling is enabled, apply predictions to unlabeled images
            if self.enabled:
                self.apply_predictions_to_unlabeled()
                # Also apply to existing auto-labeled images to refresh them with new probs
                applied_auto = 0
                repo = getattr(self.data, "_repo", None)
                for fname, prob in self.predicted_probabilities.items():
                    path = self.data.get_image_path(fname)
                    if not path:
                        continue
                    current = self.data.get_state(path)
                    source = None
                    try:
                        if repo and hasattr(repo, "get_label_source"):
                            source = repo.get_label_source(fname)
                    except Exception:
                        source = None
                    label = self.classify(prob)
                    # Refresh only auto-labeled items or unlabeled
                    if (not current) or (source == "auto"):
                        if label:
                            if self._safe_set_auto_state(path, fname, label):
                                self.auto_assigned.add(path)
                                applied_auto += 1
                            if self.callbacks.on_label_changed:
                                try:
                                    self.callbacks.on_label_changed(fname, label)
                                except Exception:
                                    logging.exception("[auto-label] on_label_changed callback failed for %s", fname)
        except Exception as e:
            logging.debug(f"[auto-label] predict_with_progress failed: {e}")

    def predict_and_apply_auto_thresholds(self) -> int:
        """Predict all unlabeled images and apply auto-labels when very-high-confidence thresholds are met.

        Returns number of labels applied.
        """
        try:
            from .training import predict_keep_probability

            if not (self.model_path and os.path.isfile(self.model_path)):
                return 0

            unlabeled, full_paths = self._get_unlabeled_images()
            if not unlabeled:
                return 0

            # Defensive: filter out any missing/None paths returned by get_image_path
            pairs = [(u, p) for u, p in zip(unlabeled, full_paths) if p]
            if not pairs:
                return 0
            unlabeled, full_paths = zip(*pairs)  # type: ignore[assignment]

            probs = predict_keep_probability(list(full_paths), model_path=self.model_path)
            applied = 0
            for fpath, prob in zip(full_paths, probs):
                if prob != prob:  # Skip NaN
                    continue
                fname = os.path.basename(fpath)
                # Track if label exists before processing
                had_label = bool(self.data.get_state(fpath))
                # Use _process_probability which handles threshold checks and auto-apply
                self._process_probability(fname, prob, auto_assign=False)
                # Check if label was applied (only count if it didn't exist before)
                if not had_label and self.data.get_state(fpath):
                    applied += 1

            if applied:
                logging.info(f"[auto-label] predict_and_apply_auto_thresholds applied={applied}")
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
            return applied
        except Exception as e:
            logging.warning(f"predict_and_apply_auto_thresholds failed: {e}")
            return 0

    def update_predictions_async(self):
        """Predict on ALL images (not just displayed) to maintain feature cache."""
        logging.info("[predict] update_predictions_async() called")
        # Skip if retraining is in progress - post-retrain workflow will handle predictions
        with self._retrain_lock:
            if self._retrain_in_progress:
                logging.info("[predict] Skipping update_predictions_async - retrain in progress")
                return
        with self._predict_lock:
            if self._predict_in_progress:
                logging.info("[predict] Already in progress - queue rerun")
                self._predict_pending = True
                return
            self._predict_in_progress = True

        logging.info("[predict] Starting prediction task")
        def _task(reporter):
            reporter.detail("predicting labels")
            try:
                from .inference import predict_keep_probability_stream

                if not (self.model_path and os.path.isfile(self.model_path)):
                    logging.warning(f"[predict] Model path invalid: {self.model_path}")
                    if self.predicted_labels:
                        self.predicted_labels.clear()
                        self.predicted_probabilities.clear()
                        if self.callbacks.on_state_snapshot:
                            try:
                                self.callbacks.on_state_snapshot()
                            except Exception:
                                pass
                    return
                
                logging.info(f"[predict] Using model: {self.model_path}")

                all_files = self.data.get_image_files()
                paths = [self.data.get_image_path(f) for f in all_files]
                # Filter out any missing paths (defensive) and keep original filenames list in sync
                filtered = [(f, p) for f, p in zip(all_files, paths) if p]
                if filtered:
                    all_files, paths = zip(*filtered)
                    all_files = list(all_files)
                    paths = list(paths)
                else:
                    all_files = []
                    paths = []
                reporter.set_total(len(paths))

                def progress_cb(current, total, detail):
                    reporter.update(current, total)
                    reporter.detail(detail)

                def per_pred_cb(fname, prob, idx, total):
                    # Skip NaN predictions - don't store them
                    if prob != prob:  # NaN check
                        return
                    # Process and record prediction (handles threshold checks and auto-apply)
                    self._process_probability(fname, prob, auto_assign=False)
                    # Trigger sorting periodically (every 10 predictions) and always at the end
                    # This ensures images are re-sorted as predictions come in, not just at the end
                    if idx % 10 == 0 or idx == total:
                        if self.callbacks.on_state_snapshot:
                            try:
                                self.callbacks.on_state_snapshot()
                            except Exception:
                                pass

                probs = predict_keep_probability_stream(
                    paths,
                    model_path=self.model_path,
                    per_prediction_callback=per_pred_cb,
                    progress_callback=progress_cb,
                )
                # Check if all predictions are NaN (broken model) - if so, clear instead of restoring
                nan_count = sum(1 for p in probs if p != p)
                if nan_count == len(probs) and len(probs) > 0:
                    logging.warning("[predict] All predictions are NaN (broken model) - clearing stale predictions")
                    self.predicted_labels.clear()
                    self.predicted_probabilities.clear()
                else:
                    # Ensure any missing entries are filled (for NaNs) - but skip NaN values
                    if len(probs) == len(all_files):
                        restored_count = 0
                        for fname, prob in zip(all_files, probs):
                            if fname not in self.predicted_probabilities and prob == prob:  # Skip NaN
                                # Use _process_probability to ensure signals are emitted
                                self._process_probability(fname, prob, auto_assign=False)
                                restored_count += 1
                logging.info(f"[predict] streaming done total={len(all_files)}")
                # Emit final state snapshot after all predictions complete to ensure all badges refresh
                # This also triggers a final re-sort by uncertainty after all predictions are loaded
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
                # Also trigger a final prediction_updated signal to ensure sorting happens
                # (even though individual predictions already triggered sorting, this ensures final sort)
                if self.callbacks.on_prediction_updated:
                    try:
                        # Emit a dummy signal to trigger final re-sort after all predictions complete
                        # The handler will re-sort all images based on current probabilities
                        self.callbacks.on_prediction_updated("", 0.0, "")  # Dummy values, just to trigger sort
                    except Exception:
                        pass
            except Exception as e:
                logging.warning(f"Predict task failed: {e}")
            finally:
                rerun = False
                with self._predict_lock:
                    self._predict_in_progress = False
                    if self._predict_pending:
                        self._predict_pending = False
                        self._predict_in_progress = True
                        rerun = True

                if rerun:
                    logging.info("[predict] Running queued rerun")
                    if self.task_runner:
                        self.task_runner.run("predict", _task)
                    else:
                        _task(None)

        if self.task_runner:
            logging.info("[predict] Running prediction task via TaskRunner")
            self.task_runner.run("predict", _task)
        else:
            logging.warning("[predict] No TaskRunner available, running inline")
            _task(None)

    # ---------------- Auto Label -----------------
    def _get_unlabeled_images(self) -> tuple[list[str], list[str]]:
        """Get lists of unlabeled filenames and their paths."""
        unlabeled = []
        full_paths = []
        image_files = self.images if self.images else self.data.get_image_files()
        for fname in image_files:
            path = self.data.get_image_path(fname)
            if path and not self.data.get_state(path):
                unlabeled.append(fname)
                full_paths.append(path)
        return unlabeled, full_paths

    def _is_manual(self, path: str, filename: str | None = None) -> bool:
        try:
            repo = getattr(self.data, "_repo", None)
            fname = filename or os.path.basename(path)
            if not repo:
                return False
            return repo.get_label_source(fname) == "manual" or fname in self.protected_manual_labels
        except Exception:
            return False

    def _safe_set_auto_state(self, path: str, filename: str, label: str) -> bool:
        """Set auto label only if existing label not manual. Returns True if changed."""
        if self._is_manual(path, filename):
            return False
        if filename in self.protected_manual_labels:
            return False
        self.data.set_state(path, label, source="auto")
        return True

    def _apply_auto_labels(self, filenames: list[str], paths: list[str], probs: list[float], reporter=None) -> int:
        """Apply auto labels based on predictions. Returns count of applied labels."""
        applied = 0
        for fname, path, p in zip(filenames, paths, probs):
            if reporter:
                reporter.detail(f"auto-label {fname}")

            if p != p:  # NaN
                if reporter:
                    reporter.advance(1)
                continue

            label = self.classify(p)
            if label and self._safe_set_auto_state(path, fname, label):
                self.auto_assigned.add(path)
                applied += 1

            if reporter:
                reporter.advance(1)

        return applied

    def _emit_label_changes(self, filenames: list[str], paths: list[str], probs: list[float]):
        """Emit label_changed callbacks for auto-labeled images."""
        if not self.callbacks.on_label_changed:
            return
        for fname, path, p in zip(filenames, paths, probs):
            if p != p:
                continue
            state = self.data.get_state(path)
            if state:
                try:
                    self.callbacks.on_label_changed(fname, state)
                except Exception:
                    logging.exception("[auto-label] on_label_changed callback failed for %s", fname)

    def apply_predictions_to_unlabeled(self) -> int:
        """Apply current predicted labels to unlabeled images (manual-safe).
        Returns number of labels applied. Skips if disabled or no predictions.
        """
        if not self.enabled or not self.predicted_probabilities:
            return 0
        applied = 0
        skipped_manual = 0
        skipped_missing = 0
        applied = 0
        repo = getattr(self.data, "_repo", None)
        for fname, prob in self.predicted_probabilities.items():
            label = self.classify(prob)
            if not label:
                continue
            path = self.data.get_image_path(fname)
            if not path:
                skipped_missing += 1
                continue
            current = self.data.get_state(path)
            if current and repo and getattr(repo, "get_label_source", lambda x: "manual")(fname) == "manual":
                skipped_manual += 1
                continue
            if not current:
                if self._safe_set_auto_state(path, fname, label):
                    self.auto_assigned.add(path)
                    applied += 1
                    if self.callbacks.on_label_changed:
                        try:
                            self.callbacks.on_label_changed(fname, label)
                        except Exception:
                            logging.exception("[auto-label] apply_predictions_to_unlabeled callback failed for %s", fname)
        if applied:
            logging.info(f"[auto-label] apply_predictions_to_unlabeled applied={applied}")
            if self.callbacks.on_state_snapshot:
                try:
                    self.callbacks.on_state_snapshot()
                except Exception:
                    pass
        return applied

    def auto_label_unlabeled_async(self):
        def _task(reporter):
            if reporter:
                reporter.detail("finding unlabeled images")
            try:
                from .inference import predict_keep_probability_stream

                if not (self.model_path and os.path.isfile(self.model_path)):
                    logging.info("Auto-label skipped: model file missing (%s)", self.model_path)
                    return
                unlabeled, full_paths = self._get_unlabeled_images()
                if reporter:
                    reporter.set_total(len(unlabeled))
                    reporter.detail(f"auto-labeling {len(unlabeled)} images")
                if not unlabeled:
                    return
                fname_to_path = {os.path.basename(p): p for p in full_paths}
                applied = 0
                
                # Always use progress reporter if available
                def per_pred_cb(fname: str, prob: float, idx: int, total: int):
                    # Use _process_probability which handles prediction storage and auto-apply
                    self._process_probability(fname, prob, auto_assign=False)

                def progress_cb(current, total, detail):
                    if reporter:
                        reporter.update(current, total)
                        reporter.detail(detail or "predicting")

                predict_keep_probability_stream(
                    full_paths,
                    model_path=self.model_path,
                    per_prediction_callback=per_pred_cb,
                    progress_callback=progress_cb,
                )
                self.apply_predictions_to_unlabeled()
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
                logging.info(
                    f"[auto-label] Streaming complete applied={applied} unlabeled_remaining={len([f for f in unlabeled if not self.data.get_state(fname_to_path[f])])}"
                )
            except Exception as e:
                logging.warning(f"Auto-label failed: {e}")
            finally:
                self.predict_with_progress()

        # Always use TaskRunner if available to show progress
        if self.task_runner:
            self.task_runner.run("auto-label", _task)
        else:
            # Fallback: create a dummy reporter for consistency
            class DummyReporter:
                def set_total(self, total): pass
                def update(self, current, total=None): pass
                def detail(self, text): pass
            _task(DummyReporter())

    # ---------------- Retraining -----------------
    def schedule_retrain(self):
        logging.info(f"[retrain] schedule_retrain called (in_progress={self._retrain_in_progress})")
        now = time.time()
        with self._retrain_lock:
            if self._retrain_in_progress:
                self._retrain_pending = True
                logging.info("[retrain] Already running - queued")
                return
        time_since_last = now - self._last_retrain_time
        delay = self._retrain_min_interval - time_since_last if time_since_last < self._retrain_min_interval else 0
        logging.info(f"[retrain] Scheduling retrain with delay={delay:.2f}s (min_interval={self._retrain_min_interval}, time_since_last={time_since_last:.2f})")

        def _delayed():
            try:
                if delay > 0:
                    logging.info(f"[retrain] Waiting {delay:.2f}s before starting retrain")
                    time.sleep(delay)
                logging.info("[retrain] Starting retrain now")
                self._start_retrain()
            except Exception as e:
                logging.exception(f"[retrain] Error in delayed retrain: {e}")
                raise
        try:
            threading.Thread(target=_delayed, daemon=True).start()
        except Exception as e:
            logging.exception(f"[retrain] Failed to start retrain thread: {e}")
            raise

    def _start_retrain(self):
        with self._retrain_lock:
            if self._retrain_in_progress:
                logging.warning("[retrain] Duplicate start ignored")
                return
            self._retrain_in_progress = True
        logging.info("[retrain] Starting background retraining")

        def _retrain_task(reporter):
            try:
                from .training import DEFAULT_MODEL_PATH, train_keep_trash_model

                model_path = self.model_path or DEFAULT_MODEL_PATH
                repo = getattr(self.data, "_repo", None)

                def progress_cb(current, total, detail):
                    try:
                        if getattr(reporter, "total", None) != total and total:
                            reporter.set_total(total)
                        reporter.update(current, total)
                        reporter.detail(detail)
                    except Exception:
                        logging.exception("Error in auto-label assignment")
                        raise

                # Introspect signature to maintain compatibility with test stubs not updated yet
                import inspect

                sig = inspect.signature(train_keep_trash_model)
                # Build call accommodating legacy signature without 'image_dir'
                if "image_dir" in sig.parameters:
                    kwargs = dict(
                        image_dir=self.data.directory,
                        model_path=model_path,
                        repo=repo,
                        n_estimators=120,
                        progress_callback=progress_cb,
                        fast_mode=True,  # Use fast mode for app retraining (<5s target)
                    )
                    result = train_keep_trash_model(**kwargs)  # type: ignore[arg-type]
                else:
                    # Legacy positional: (directory, model_path, random_state, repo, n_estimators, progress_callback)
                    result = train_keep_trash_model(self.data.directory, model_path, 42, repo, 120, progress_cb)  # type: ignore[misc]
                if result:
                    self.model_stats = result
                    # Update model_path to use the newly trained model
                    if result.model_path and os.path.isfile(result.model_path):
                        self.model_path = result.model_path
                        logging.info(f"[retrain] Updated model_path to {self.model_path}")
                        # Invalidate model cache to force reload of new model
                        from .inference import invalidate_model_cache

                        invalidate_model_cache()  # Clear all caches to ensure fresh model is loaded
                    # Compute effective (possibly weighted) thresholds
                    keep_eff, trash_eff = self.compute_dynamic_thresholds()
                    stats_dict = {
                        "n_samples": result.n_samples,
                        "n_keep": result.n_keep,
                        "n_trash": result.n_trash,
                        "cv_accuracy_mean": result.cv_accuracy_mean,
                        "cv_accuracy_std": result.cv_accuracy_std,
                        "precision": result.precision,
                        "model_path": result.model_path,
                        "roc_auc": result.roc_auc,
                        "f1": result.f1,
                        # Threshold details
                        "keep_threshold_base": self.keep_thr,
                        "trash_threshold_base": self.trash_thr,
                        "keep_threshold_eff": keep_eff,
                        "trash_threshold_eff": trash_eff,
                        "weighted_mode": self.weighted_mode,
                    }
                    # Include per-epoch training history if available
                    try:
                        if getattr(result, "training_history", None):
                            stats_dict["training_history"] = result.training_history
                    except Exception:
                        pass
                    # Attempt to load persisted metrics and feature importances for UI display
                    try:
                        import joblib

                        if os.path.isfile(result.model_path):
                            data = joblib.load(result.model_path)
                            # Load feature importances
                            fi = data.get("feature_importances") or getattr(result, "feature_importances", None)
                            if fi:
                                stats_dict["feature_importances"] = fi
                            # Load F1 score (may not be in result if model was loaded from disk)
                            f1 = data.get("f1")
                            if f1 is not None:
                                stats_dict["f1"] = f1
                            # Load other metrics if not already in result
                            if "roc_auc" not in stats_dict:
                                roc_auc = data.get("roc_auc")
                                if roc_auc is not None:
                                    stats_dict["roc_auc"] = roc_auc
                            if "precision" not in stats_dict:
                                precision = data.get("precision")
                                if precision is not None:
                                    stats_dict["precision"] = precision
                            # Add model metadata for feature name mapping (interactions, embeddings)
                            metadata = data.get("__metadata__")
                            if metadata:
                                stats_dict["model_metadata"] = metadata
                    except Exception:
                        logging.exception("Error in auto-label scoring")
                        raise
                    if self.callbacks.on_model_stats_changed:
                        try:
                            self.callbacks.on_model_stats_changed(stats_dict)
                        except Exception:
                            logging.exception("[auto-label] on_model_stats_changed callback failed")
                else:
                    logging.info("[retrain] Skipped (insufficient data)")
            except Exception as e:
                logging.warning(f"Retrain failed: {e}", exc_info=True)
            finally:
                with self._retrain_lock:
                    self._retrain_in_progress = False
                    self._last_retrain_time = time.time()
                    pending = self._retrain_pending
                    self._retrain_pending = False
                if pending:
                    logging.info("[retrain] Running queued retrain")
                    self._start_retrain()
                # Post-retrain workflow
                try:
                    reporter.detail("predicting after retrain")
                    logging.info("[retrain] Starting predictions with new model")
                    self.predict_with_progress()
                    logging.info("[retrain] Predictions complete")
                    if self.enabled:
                        logging.info("[retrain] Refreshing auto-labels (enabled)")
                        reporter.detail("refreshing auto-labels")
                        self.refresh_auto_labels()
                    else:
                        logging.info(
                            "[retrain] Auto-labeling disabled; skipping auto-label refresh but predictions are updated"
                        )
                    logging.info("[retrain] Post-retrain workflow complete, emitting state snapshot")
                    if self.callbacks.on_state_snapshot:
                        try:
                            self.callbacks.on_state_snapshot()
                        except Exception:
                            pass
                except Exception as e:
                    logging.warning(f"Post-retrain workflow failed: {e}", exc_info=True)
        try:
            if self.task_runner:
                self.task_runner.run("retrain", _retrain_task)
            else:
                _retrain_task(None)
        except Exception as e:
            raise

    # ---------------- Auto Label Refresh / Apply -----------------
    def _rebuild_auto_assigned_tracking(self) -> set[str]:
        """Rebuild auto_assigned set from repository."""
        rebuilt: set[str] = set()
        repo = getattr(self.data, "_repo", None)
        if not repo:
            return rebuilt

        image_files = self.images if self.images else self.data.get_image_files()
        for fname in image_files:
            path = self.data.get_image_path(fname)
            if not path:
                continue
            try:
                if repo.get_label_source(fname) == "auto" and self.data.get_state(path):
                    rebuilt.add(path)
            except Exception:
                logging.exception("Error batching auto-label predictions")
                raise
        return rebuilt

    def _update_auto_label(
        self, path: str, filename: str, predicted: str | None, current: str, changed_files: list, removed_files: list
    ) -> tuple[int, int]:
        """Update a single auto label. Returns (changed_count, removed_count).
        Only operates on auto-labeled images; manual labels are preserved.
        """
        if predicted is None:
            return 0, 0
        # Determine current source
        try:
            repo = getattr(self.data, "_repo", None)
            current_source = repo.get_label_source(filename) if repo else "manual"
        except Exception:
            current_source = "manual"
        # Ignore if manual label
        if current_source == "manual":
            return 0, 0
        if filename in self.protected_manual_labels:
            return 0, 0
        # Removal if prediction now empty
        if not predicted:
            if current:
                if not self._is_manual(path, filename):
                    self.data.set_state(path, "", source="auto")
                    self.auto_assigned.discard(path)
                    removed_files.append((filename, current))
                    return 0, 1
            return 0, 0
        # Change auto label if differs
        if current != predicted:
            if self._safe_set_auto_state(path, filename, predicted):
                changed_files.append((filename, current, predicted))
                return 1, 0
        return 0, 0

    def refresh_auto_labels(self):
        if not self.predicted_labels:
            return
        
        # Run in background with progress if TaskRunner available
        if self.task_runner:
            def _task(reporter):
                # Rebuild tracking if empty (supports test expectations)
                if not self.auto_assigned:
                    rebuilt = self._rebuild_auto_assigned_tracking()
                    if rebuilt:
                        self.auto_assigned = rebuilt
                
                auto_assigned_list = list(self.auto_assigned)
                reporter.set_total(len(auto_assigned_list))
                reporter.detail("refreshing auto-labels")
                
                changed = 0
                removed = 0
                changed_files = []
                removed_files = []
                
                for idx, path in enumerate(auto_assigned_list):
                    filename = os.path.basename(path)
                    if filename in self.protected_manual_labels:
                        continue
                    predicted = self.predicted_labels.get(filename)
                    current = self.data.get_state(path)
                    c, r = self._update_auto_label(path, filename, predicted, current, changed_files, removed_files)
                    changed += c
                    removed += r
                    reporter.update(idx + 1)
                    reporter.detail(f"checked {idx + 1}/{len(auto_assigned_list)}")

                if changed:
                    logging.info(f"[refresh_auto_labels] Updated {changed} auto-labeled images after retrain")
                    if self.callbacks.on_label_changed:
                        for fname, old_label, new_label in changed_files:
                            try:
                                self.callbacks.on_label_changed(fname, new_label)
                            except Exception:
                                logging.exception("[auto-label] on_label_changed callback failed for %s", fname)

                if removed:
                    logging.info(f"[refresh_auto_labels] Removed {removed} auto-labels (predictions below threshold)")
                    if self.callbacks.on_label_changed:
                        for fname, old_label in removed_files:
                            try:
                                self.callbacks.on_label_changed(fname, "")
                            except Exception:
                                logging.exception("[auto-label] on_label_changed callback failed for %s", fname)

                if changed or removed:
                    if self.callbacks.on_state_snapshot:
                        try:
                            self.callbacks.on_state_snapshot()
                        except Exception:
                            pass
                else:
                    logging.debug(f"[refresh_auto_labels] No auto-label changes needed ({len(auto_assigned_list)} checked)")
            
            self.task_runner.run("refresh-auto-labels", _task)
        else:
            # Synchronous fallback
            # Rebuild tracking if empty (supports test expectations)
            if not self.auto_assigned:
                rebuilt = self._rebuild_auto_assigned_tracking()
                if rebuilt:
                    self.auto_assigned = rebuilt
            changed = 0
            removed = 0
            changed_files = []
            removed_files = []
            for path in list(self.auto_assigned):
                filename = os.path.basename(path)
                if filename in self.protected_manual_labels:
                    continue
                predicted = self.predicted_labels.get(filename)
                current = self.data.get_state(path)
                c, r = self._update_auto_label(path, filename, predicted, current, changed_files, removed_files)
                changed += c
                removed += r

            if changed:
                logging.info(f"[refresh_auto_labels] Updated {changed} auto-labeled images after retrain")
                if self.callbacks.on_label_changed:
                    for fname, old_label, new_label in changed_files:
                        try:
                            self.callbacks.on_label_changed(fname, new_label)
                        except Exception:
                            logging.exception("[auto-label] on_label_changed callback failed for %s", fname)

            if removed:
                logging.info(f"[refresh_auto_labels] Removed {removed} auto-labels (predictions below threshold)")
                if self.callbacks.on_label_changed:
                    for fname, old_label in removed_files:
                        try:
                            self.callbacks.on_label_changed(fname, "")
                        except Exception:
                            logging.exception("[auto-label] on_label_changed callback failed for %s", fname)

            if changed or removed:
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
            else:
                logging.debug(f"[refresh_auto_labels] No auto-label changes needed ({len(self.auto_assigned)} checked)")

    def confirm_review_label(self, filename: str, user_label: str) -> bool:
        """User confirms or corrects an auto-label prediction.

        Args:
            filename: Image filename
            user_label: 'keep', 'trash', or '' to reject

        Returns: True if successful
        """
        if not self.review_manager:
            return False
        return self.review_manager.confirm_label(filename, user_label)  # type: ignore[no-any-return, attr-defined]

    def apply_confirmed_labels_to_repository(self) -> int:
        """Apply all user-confirmed labels to the repository.

        Returns: Number of labels added to repository
        """
        repo = getattr(self.data, "_repo", None)
        if not self.review_manager or not repo:
            return 0

        added_raw = self.review_manager.add_confirmed_to_repository(repo)  # type: ignore[attr-defined]
        added = int(added_raw) if added_raw else 0  # type: ignore[arg-type]

        if added > 0:
            logging.info(f"[auto-label] Applied {added} confirmed labels to repository")
            self.review_manager.clear_reviewed()  # type: ignore[attr-defined]
            if self.callbacks.on_state_snapshot:
                try:
                    self.callbacks.on_state_snapshot()
                except Exception:
                    pass

        return added

    def set_thresholds(self, keep: float, trash: float) -> bool:
        """Dynamically update auto-label thresholds.
        Returns True if applied. Re-validates and re-predicts with new thresholds.
        """
        try:
            # Validate threshold sanity
            is_valid = keep > trash and 0 < keep <= 1 and 0 <= trash < 1

            if not is_valid:
                logging.warning(f"[auto-label] Invalid thresholds keep={keep} trash={trash}")
                return False

            self.keep_thr = round(float(keep), 4)
            self.trash_thr = round(float(trash), 4)
            logging.info(f"[auto-label] Thresholds updated keep={self.keep_thr} trash={self.trash_thr}")

            # Re-map existing predicted labels based on new thresholds
            if self.predicted_probabilities:
                for fname, prob in self.predicted_probabilities.items():
                    new_label = self.classify(prob)
                    old_label = self.predicted_labels.get(fname)
                    if new_label != old_label:
                        self.predicted_labels[fname] = new_label
                        if new_label and self.callbacks.on_prediction_updated:
                            try:
                                self.callbacks.on_prediction_updated(fname, prob, new_label)
                            except Exception:
                                logging.exception("[auto-label] on_prediction_updated callback failed for %s", fname)
                if self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
            return True
        except Exception as e:
            logging.warning(f"[auto-label] set_thresholds failed: {e}")
            return False

    def ensure_initial_labels(self):
        """Assign labels to any currently unlabeled images that have probabilities but no state yet."""
        if not self.predicted_probabilities:
            return 0
        
        # Run in background with progress if TaskRunner available
        if self.task_runner:
            def _task(reporter):
                unlabeled_items = [(fname, prob) for fname, prob in self.predicted_probabilities.items() 
                                 if not self.data.get_state(self.data.get_image_path(fname) or "")]
                reporter.set_total(len(unlabeled_items))
                reporter.detail("applying initial labels")
                applied = 0
                for idx, (fname, prob) in enumerate(unlabeled_items):
                    path = self.data.get_image_path(fname)
                    if not path:
                        continue
                    label = self.classify(prob)
                    if label and self._safe_set_auto_state(path, fname, label):
                        self.auto_assigned.add(path)
                        applied += 1
                        if self.callbacks.on_label_changed:
                            try:
                                self.callbacks.on_label_changed(fname, label)
                            except Exception:
                                logging.exception("[auto-label] on_label_changed callback failed for %s", fname)
                    reporter.update(idx + 1)
                    reporter.detail(f"applied {applied} labels")
                if applied and self.callbacks.on_state_snapshot:
                    try:
                        self.callbacks.on_state_snapshot()
                    except Exception:
                        pass
            
            # Run async with progress
            self.task_runner.run("ensure-initial-labels", _task)
            return 0  # Return immediately, actual count handled in background
        else:
            # Synchronous fallback
            applied = 0
            for fname, prob in self.predicted_probabilities.items():
                path = self.data.get_image_path(fname)
                if not path or self.data.get_state(path):
                    continue
                label = self.classify(prob)
                if label and self._safe_set_auto_state(path, fname, label):
                    self.auto_assigned.add(path)
                    applied += 1
                    if self.callbacks.on_label_changed:
                        try:
                            self.callbacks.on_label_changed(fname, label)
                        except Exception:
                            logging.exception("[auto-label] on_label_changed callback failed for %s", fname)
            if applied and self.callbacks.on_state_snapshot:
                try:
                    self.callbacks.on_state_snapshot()
                except Exception:
                    pass
            return applied
