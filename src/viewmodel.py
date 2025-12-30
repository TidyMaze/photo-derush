import logging
import os
import time

from PySide6.QtCore import QObject, Signal

from . import object_detection
from .auto_label_manager import AutoLabelManager  # type: ignore
from .cache import ThumbnailCache  # type: ignore
from .commands import CommandStack, SetLabelCommand  # type: ignore
from .detection_worker import stop_global_worker  # type: ignore
from .domain import ImageBrowserState  # type: ignore
from .model import ImageModel, RatingsTagsRepository  # type: ignore
from .selection import SelectionModel  # type: ignore
from .services import ExifService, ThumbnailService  # type: ignore
from .taskrunner import TaskRunner  # type: ignore
from .vm_filters import FilterController  # type: ignore
from .vm_labeling import LabelController  # type: ignore
from .grouping_service import compute_grouping_for_photos  # type: ignore


class PhotoViewModel(QObject):
    """ViewModel: Exposes all state and actions for the View. No UI code."""

    images_changed = Signal(list)
    image_added = Signal(str, int)
    exif_changed = Signal(dict)
    progress_changed = Signal(int, int)
    thumbnail_loaded = Signal(str, object)
    selected_image_changed = Signal(str)
    has_selected_image_changed = Signal(bool)
    rating_changed = Signal(int)
    tags_changed = Signal(list)
    label_changed = Signal(str, str)
    prediction_updated = Signal(str, float, str)  # filename, probability, label (for incremental updates)
    browser_state_changed = Signal(object)  # emits ImageBrowserState
    selection_changed = Signal(list)  # list of selected full paths
    primary_selection_changed = Signal(str)  # full path
    undo_redo_changed = Signal(bool, bool)  # can_undo, can_redo
    task_started = Signal(str)
    task_progress = Signal(str, int, int, str)
    task_finished = Signal(str, bool)
    model_stats_changed = Signal(dict)  # emits model performance metrics
    directory_changed = Signal(str)  # emits new directory path
    object_detection_ready = Signal(str)  # filename when detection completes
    grouping_completed = Signal()  # emitted when grouping computation finishes

    def __init__(self, directory, max_images=10000):
        super().__init__()
        self._init_model(directory, max_images)
        self._init_state()
        self._init_services()
        self._init_tasks()  # Initialize tasks before controllers (AutoLabelManager needs task_runner)
        self._init_controllers()
        self._connect_signals()

    def _init_model(self, directory, max_images):
        cache = ThumbnailCache()
        repo_path = os.path.join(directory, ".ratings_tags.json")
        try:
            scoped_repo = RatingsTagsRepository(path=repo_path)
        except Exception:
            scoped_repo = None
        self.model = ImageModel(directory, max_images, cache=cache, repo=scoped_repo)
        self.max_images = max_images

    def _init_state(self):
        self.images = []
        self.selected_image: str | None = None
        self.exif = {}
        self._has_selected_image: bool = False
        self._rating = 0
        self._tags = []
        self._label = None
        self._detected_objects = {}
        self._pending_label_changes = {}
        self._progress_current = 0
        self._progress_total = 0
        self._loading_detections = False
        self._detection_task_running = False
        self._auto_selected_most_boxes = False
        self._group_info: dict[str, dict] = {}  # filename -> grouping info
        self._grouping_computed = False

        import threading
        self._retrain_lock = threading.Lock()
        self._retrain_in_progress = False
        self._retrain_pending = False
        self._last_retrain_time = 0.0
        self._nan_retrain_attempts = 0

    def _init_services(self):
        self.thumbnail_service = ThumbnailService(self.model)
        self.exif_service = ExifService(self.model)
        self._thumb_requested = set()
        self._exif_cache = {}
        self._exif_inflight = set()

    def _init_controllers(self):
        self.selection_model = SelectionModel()
        self._cmd_stack = CommandStack()
        self._filter_ctrl = FilterController()
        self._filtered_images = []
        # Initialize AutoLabelManager with brain interface
        from .brain_interface import BrainAdapter, BrainCallbacks

        data_provider = BrainAdapter(self.model)
        
        def safe_emit_prediction(fname: str, prob: float, label: str):
            """Safely emit prediction signal, ignoring errors if signal source deleted."""
            try:
                self.prediction_updated.emit(fname, prob, label)
            except RuntimeError as e:
                if "Signal source has been deleted" in str(e):
                    # UI closed or object deleted - harmless, ignore
                    pass
                else:
                    raise
        
        def safe_emit_label(fname: str, state: str):
            """Safely emit label signal, ignoring errors if signal source deleted."""
            try:
                self.label_changed.emit(fname, state)
            except RuntimeError as e:
                if "Signal source has been deleted" in str(e):
                    # UI closed or object deleted - harmless, ignore
                    pass
                else:
                    raise
        
        def safe_emit_stats(stats: dict):
            """Safely emit stats signal, ignoring errors if signal source deleted."""
            try:
                self.model_stats_changed.emit(stats)
            except RuntimeError as e:
                if "Signal source has been deleted" in str(e):
                    # UI closed or object deleted - harmless, ignore
                    pass
                else:
                    raise
        
        callbacks = BrainCallbacks(
            on_label_changed=safe_emit_label,
            on_prediction_updated=safe_emit_prediction,
            on_state_snapshot=self._emit_state_snapshot,
            on_model_stats_changed=safe_emit_stats,
        )
        self._auto = AutoLabelManager(
            data_provider=data_provider,
            callbacks=callbacks,
            task_runner=self._tasks,
            images=None,  # Will use data_provider.get_image_files() when needed
        )
        self._label_ctrl = LabelController(self)

    def _init_tasks(self):
        self._tasks = TaskRunner()
        self._active_tasks = set()

    def _connect_signals(self):
        self.selection_model.selectionChanged.connect(self._on_selection_changed)
        self.selection_model.primaryChanged.connect(self._on_primary_changed)
        self._tasks.task_started.connect(self.task_started.emit)
        self._tasks.task_progress.connect(self.task_progress.emit)
        self._tasks.task_finished.connect(self.task_finished.emit)
        self._tasks.task_started.connect(self._on_task_started)
        self._tasks.task_finished.connect(self._on_task_finished)
        self.grouping_completed.connect(self._on_grouping_completed)
        # Re-sort whenever predictions change
        self.prediction_updated.connect(self._on_prediction_updated)
        # Re-sort whenever labels change
        self.label_changed.connect(self._on_label_changed)
        self._emit_undo_redo()
        self._apply_filters()

        # ---------------- Manual Training (non-blocking) -----------------
        self._training_cancel_requested = False
        self._training_active: bool = False

        # Debounce state snapshots to avoid excessive UI updates
        from PySide6.QtCore import QTimer
        self._snapshot_timer = QTimer()
        self._snapshot_timer.setSingleShot(True)
        self._snapshot_timer.timeout.connect(self._emit_state_snapshot_immediate)
        self._snapshot_pending = False
        self._snapshot_debounce_ms = 200  # Debounce to max 5 calls/sec
        
        # PERFORMANCE: Debounce _apply_filters to reduce excessive calls (1,196 calls -> ~10/sec max)
        self._apply_filters_timer = QTimer()
        self._apply_filters_timer.setSingleShot(True)
        self._apply_filters_pending = False

        # Note: don't emit a state snapshot from here — _emit_state_snapshot
        # calls _load_object_detections, which would cause infinite recursion.
        # The initial UI snapshot is emitted elsewhere after initialization.

    def _load_object_detections(self):
        """Load object detections for current images."""
        if getattr(self, "_loading_detections", False):
            return
        if not self.images:
            self._detected_objects = {}
            return
        self._loading_detections = True
        try:
            # Cache the loaded cache to avoid repeated disk I/O
            cache_path = object_detection.get_cache_path()
            cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
            cached_cache = getattr(self, "_cached_detection_cache", None)
            cached_mtime = getattr(self, "_cached_detection_cache_mtime", 0)
            
            # Always reload if cache file was modified (new detections available)
            if cached_cache is None or cache_mtime != cached_mtime:
                cache = object_detection.load_object_cache()
                self._cached_detection_cache = cache
                self._cached_detection_cache_mtime = cache_mtime
            else:
                cache = cached_cache
            
            # Pre-compute basename mappings to avoid repeated os.path.basename calls
            if not hasattr(self, "_image_basename_cache") or len(self._image_basename_cache) != len(self.images):
                self._image_basename_cache = {}
                for fname in self.images:
                    p_candidate = self.model.get_image_path(fname)
                    if not p_candidate:
                        p_candidate = os.path.join(self.model.directory, fname)
                    self._image_basename_cache[fname] = os.path.basename(p_candidate)
            
            results: dict = {}
            to_process: list[str] = []

            cache_hits = 0
            cache_misses = 0
            for fname in self.images:
                base = self._image_basename_cache[fname]
                if base in cache:
                    # cache stores list[dict]
                    results[base] = cache[base]
                    cache_hits += 1
                else:
                    # ensure we have an entry (empty until detection completes)
                    results[base] = []
                    cache_misses += 1
                    p_candidate = self.model.get_image_path(fname)
                    if not p_candidate:
                        p_candidate = os.path.join(self.model.directory, fname)
                    if p_candidate:
                        to_process.append(p_candidate)

            # Store current snapshot of available detections (fast, non-blocking)
            # Only load from cache synchronously to avoid blocking UI thread.
            # Background detection task will handle missing detections.
            self._detected_objects = results

            # Schedule background detection for missing images to avoid blocking UI
            if len(to_process) > 0:
                logging.debug(f"[DETECTION] {len(to_process)} images need detection, task_running={getattr(self, '_detection_task_running', False)}")
            # Only start detection task if there are images to process AND no task is already running
            # This prevents duplicate tasks from being started when _load_object_detections is called frequently
            if to_process and not getattr(self, "_detection_task_running", False):

                def _detect_task(reporter):
                    # mark running
                    self._detection_task_running = True
                    try:
                        if reporter:
                            reporter.detail("loading object detection cache")
                            reporter.set_total(len(to_process))
                        
                        # Use cached cache if available and still valid, otherwise reload
                        cache_path = object_detection.get_cache_path()
                        cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                        cached_cache = getattr(self, "_cached_detection_cache", None)
                        cached_mtime = getattr(self, "_cached_detection_cache_mtime", 0)
                        
                        if cached_cache is not None and cache_mtime == cached_mtime:
                            c = cached_cache.copy()  # Use cached copy
                        else:
                            # Reload if cache changed or not cached
                            c = object_detection.load_object_cache()
                            self._cached_detection_cache = c
                            self._cached_detection_cache_mtime = cache_mtime

                        # Always use detect_objects directly to get bbox data
                        # get_objects_for_images returns tuples without bboxes, so skip it
                        if reporter:
                            reporter.detail("detecting objects with bboxes")
                        for idx, path in enumerate(to_process):
                            base = os.path.basename(path)
                            # Check cache again (might have been added by another thread or already cached)
                            if base in c:
                                # Use cached result
                                self._detected_objects[base] = c[base]
                                self.object_detection_ready.emit(base)
                                if reporter:
                                    reporter.update(idx + 1, len(to_process))
                                    reporter.detail(f"cached {base}")
                                continue
                            
                            dets = []
                            # Prefer full detections (with bbox) by calling detect_objects
                            # This runs in the background task so it's acceptable to call
                            # the potentially heavy detection routine here. It ensures
                            # the cache contains bounding-box coordinates for UI overlay.
                            dets = object_detection.detect_objects(path)
                            # Ensure all dict entries are canonical via sanitizer
                            new_dets = []
                            for d in dets:
                                new_dets.append(object_detection.sanitize_detection(d))
                            c[base] = new_dets
                            self._detected_objects[base] = new_dets
                            self.object_detection_ready.emit(base)
                            
                            # Save cache immediately after each detection to prevent data loss
                            try:
                                object_detection.save_object_cache(c)
                            except Exception as e:
                                logging.warning(f"Failed to save cache after detection for {base}: {e}")
                            
                            if reporter:
                                reporter.update(idx + 1, len(to_process))
                                reporter.detail(f"detected {base}")

                        try:
                            # Cache is now saved after each detection, so no need to save at the end
                            # Update in-memory cache to reflect new detections (don't invalidate - keep in memory)
                            cache_path = object_detection.get_cache_path()
                            self._cached_detection_cache = c
                            self._cached_detection_cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
                        except Exception:
                            logging.exception("Error in viewmodel cache invalidation")
                            # Don't raise - this is not critical

                        # Refresh self._detected_objects from updated cache
                        new_results = {}
                        for fname in self.images:
                            base = os.path.basename(self.model.get_image_path(fname) or fname)
                            new_results[base] = c.get(base, [])
                        self._detected_objects = new_results
                        # Emit a fresh snapshot so UI updates (safe to call)
                        try:
                            self._emit_state_snapshot()
                        except Exception:
                            logging.exception("Error in viewmodel formatting")
                            raise
                    finally:
                        self._detection_task_running = False

                # run detection in background task runner
                try:
                    self._tasks.run("detect-objects", _detect_task)
                except Exception:
                    # Fallback: run inline but keep UI safe
                    _detect_task()
        except Exception as e:
            # Propagate ValueError (sanitizer fail-fast) to callers/tests.
            if isinstance(e, ValueError):
                raise
            logging.warning(f"Failed to load object detections: {e}")
            self._detected_objects = {}
        finally:
            self._loading_detections = False

    @property
    def has_selected_image(self) -> bool:
        return self._has_selected_image

    def _on_task_started(self, task_name: str):
        self._active_tasks.add(task_name)

    def _on_task_finished(self, task_name: str, success: bool):
        self._active_tasks.discard(task_name)
        if not self._is_blocking_task_running() and self._pending_label_changes:
            self._apply_pending_label_changes()

    def _is_blocking_task_running(self) -> bool:
        return bool(self._active_tasks & {"retrain", "predict", "auto-label"})

    @property
    def rating(self):
        return self._rating

    @property
    def tags(self):
        return self._tags

    @property
    def label(self):
        return self._label

    @property
    def can_undo(self):
        return bool(self._cmd_stack and self._cmd_stack.can_undo)

    @property
    def can_redo(self):
        return bool(self._cmd_stack and self._cmd_stack.can_redo)

    def _emit_undo_redo(self):
        if self._cmd_stack:
            self.undo_redo_changed.emit(self._cmd_stack.can_undo, self._cmd_stack.can_redo)

    def _apply_filters(self):
        # PERFORMANCE: Clear filter caches when filters change to ensure fresh data
        # This prevents stale cache entries while still benefiting from repeated lookups
        if not hasattr(self, "_last_filter_state") or self._last_filter_state != (
            self._filter_ctrl.rating, self._filter_ctrl.tag, self._filter_ctrl.date, self._filter_ctrl.hide_manual
        ):
            self._filter_path_cache = {}
            self._filter_state_cache = {}
            self._last_filter_state = (
                self._filter_ctrl.rating, self._filter_ctrl.tag, self._filter_ctrl.date, self._filter_ctrl.hide_manual
            )
        
        rating_req = int(self._filter_ctrl.rating or 0)
        tag_req = (self._filter_ctrl.tag or "").lower().strip()
        date_req = (self._filter_ctrl.date or "").strip()
        hide_manual = self._filter_ctrl.hide_manual
        
        # If no filters are active, include all images (will still be sorted by uncertainty)
        if not rating_req and not tag_req and not date_req and not hide_manual:
            filtered = list(self.images)
        else:
            filtered = []
            repo = getattr(self.model, "_repo", None)
            # Pre-compute paths to avoid repeated lookups
            if not hasattr(self, "_image_path_cache") or len(self._image_path_cache) != len(self.images):
                self._image_path_cache = {}
                for fname in self.images:
                    path = self.model.get_image_path(fname)
                    if path:
                        self._image_path_cache[fname] = path
            
            for fname in self.images:
                path = self._image_path_cache.get(fname)
                if not path:
                    continue

                # Hide manually labeled images if filter is enabled
                if hide_manual:
                    try:
                        if repo:
                            source = repo.get_label_source(fname)
                            state = repo.get_state(fname)
                            # Only hide if explicitly marked as manual AND has a label
                            if source == "manual" and state:
                                logging.debug(f"[filters] Hiding manual labeled image: {fname} (state={state})")
                                continue
                        else:
                            logging.debug(f"[filters] No repo available for {fname}, cannot check label source")
                    except Exception as e:
                        logging.debug(f"[filters] Failed to get label source for {fname}: {e}")
                        pass  # If repo check fails, include the image

                r = self.model.get_rating(path)
                if rating_req and r < rating_req:
                    continue

                if tag_req:
                    tags_lower = [t.lower() for t in self.model.get_tags(path)]
                    if tag_req not in tags_lower:
                        continue

                if date_req:
                    exif = self.model.load_exif(path)
                    exif_date = exif.get("DateTimeOriginal") or exif.get("DateTime") or ""
                    if not exif_date.startswith(date_req):
                        continue

                filtered.append(fname)
        
        # Sort by group: images in same group together, groups sorted by date ASC
        # Within each group, sort by pick_score DESC (best picks first)
        
        from datetime import datetime
        import os
        
        # Helper to extract timestamp from image
        def get_image_timestamp(fname: str) -> datetime:
            path = self._image_path_cache.get(fname) if hasattr(self, "_image_path_cache") else self.model.get_image_path(fname)
            if path:
                try:
                    exif = self.model.load_exif(path)
                    dt_original = exif.get("DateTimeOriginal")
                    if dt_original and isinstance(dt_original, str):
                        try:
                            return datetime.strptime(dt_original, "%Y:%m:%d %H:%M:%S")
                        except Exception:
                            return datetime.fromtimestamp(os.path.getmtime(path))
                    else:
                        return datetime.fromtimestamp(os.path.getmtime(path))
                except Exception:
                    return datetime.now()
            return datetime.now()
        
        # Get group info (may be empty if grouping not computed yet)
        group_info = getattr(self, "_group_info", {})
        
        # If no group info available, fall back to uncertainty sorting
        if not group_info:
            # Fallback: sort by uncertainty (original behavior)
            probs = dict(getattr(self._auto, "predicted_probabilities", {}))
            import math
            log2 = math.log(2)
            uncertainty_cache = {}
            for fname in filtered:
                prob = probs.get(fname)
                if prob is None:
                    uncertainty = 1.0
                else:
                    prob_clamped = max(0.0001, min(0.9999, prob))
                    entropy = -(prob_clamped * math.log(prob_clamped) + (1 - prob_clamped) * math.log(1 - prob_clamped)) / log2
                    uncertainty = entropy
                uncertainty_cache[fname] = uncertainty
            
            def uncertainty_score(fname: str) -> tuple[float, str]:
                uncertainty = uncertainty_cache.get(fname, 1.0)
                return (uncertainty, fname)
            
            filtered.sort(key=uncertainty_score, reverse=True)
        else:
            # Compute earliest date per group (for sorting groups)
            group_earliest_date: dict[int | None, datetime] = {}
            
            # Get timestamps and track earliest date per group
            for fname in filtered:
                ts = get_image_timestamp(fname)
                ginfo = group_info.get(fname, {})
                group_id = ginfo.get("group_id")
                if group_id not in group_earliest_date or ts < group_earliest_date[group_id]:
                    group_earliest_date[group_id] = ts
            
            # Sort key: (group_earliest_date, -pick_score, filename)
            # Groups sorted by earliest date ASC, within group by pick_score DESC
            def group_sort_key(fname: str) -> tuple[datetime, float, str]:
                ginfo = group_info.get(fname, {})
                group_id = ginfo.get("group_id")
                pick_score = ginfo.get("pick_score", 0.0)
                # Use earliest date of the group, or image's own date if group_id is None
                if group_id is not None and group_id in group_earliest_date:
                    group_date = group_earliest_date[group_id]
                else:
                    # Fallback: use image's own date
                    group_date = get_image_timestamp(fname)
                # Use negative pick_score for descending sort within group
                return (group_date, -pick_score, fname)
            
            # Sort by group (date ASC) and pick_score (DESC within group)
            filtered.sort(key=group_sort_key)
            
            # Debug: log first few items to verify sorting
            if len(filtered) > 0:
                logging.info(f"[sorting] First 5 items after group sort:")
                for i, fname in enumerate(filtered[:5]):
                    ginfo = group_info.get(fname, {})
                    gid = ginfo.get("group_id")
                    ps = ginfo.get("pick_score", 0.0)
                    gdate = group_earliest_date.get(gid, datetime.now())
                    logging.info(f"[sorting]   {i+1}. {fname[:40]:40s} group={gid} date={gdate} score={ps:.4f}")
        
        # Check if selected image is filtered out, and mark for selection update
        selected_was_filtered = False
        new_selected_filename = None
        if self.selected_image:
            # Get filename from selected path
            selected_filename = None
            for fname in self.images:
                path = self._image_path_cache.get(fname) if hasattr(self, "_image_path_cache") else self.model.get_image_path(fname)
                if path == self.selected_image:
                    selected_filename = fname
                    break
            
            if selected_filename:
                # Check if selected image is still in filtered list
                if selected_filename not in filtered:
                    selected_was_filtered = True
                    # Find the position of the selected image in the previous filtered list
                    prev_filtered = getattr(self, "_filtered_images", [])
                    try:
                        prev_index = prev_filtered.index(selected_filename)
                    except ValueError:
                        # Selected image wasn't in previous filtered list, use index 0
                        prev_index = 0
                    
                    # Select the closest image (same index, or closest available)
                    if filtered and prev_index < len(filtered):
                        # Select image at same position
                        new_selected_filename = filtered[prev_index]
                    elif filtered:
                        # Select first image if previous index is out of bounds
                        new_selected_filename = filtered[0]
                    # else: No images left after filtering, new_selected_filename stays None
        
        self._filtered_images = filtered
        
        # Update selection if needed (after setting _filtered_images to avoid recursion)
        if selected_was_filtered and new_selected_filename:
            new_path = self._image_path_cache.get(new_selected_filename) if hasattr(self, "_image_path_cache") else self.model.get_image_path(new_selected_filename)
            if new_path:
                # Use selection_model directly to avoid calling select_image (which would call _apply_filters again)
                self.selection_model.replace(new_path)
                primary_changed = self.selected_image != new_path
                self.selected_image = new_path
                self._has_selected_image = True
                if primary_changed:
                    self.selected_image_changed.emit(new_path)
                    self.has_selected_image_changed.emit(True)
                self._load_details_for_filename(new_selected_filename)
        
        # Ensure at least one image is always selected
        self._ensure_selection()

    def set_filter_rating(self, rating: int):
        if self._filter_ctrl.set_rating(rating):
            self._apply_filters()
            self._emit_state_snapshot()

    def set_filter_tag(self, tag: str):
        if self._filter_ctrl.set_tag(tag):
            self._apply_filters()
            self._emit_state_snapshot()

    def set_filter_date(self, date: str):
        if self._filter_ctrl.set_date(date):
            self._apply_filters()
            self._emit_state_snapshot()

    def set_filter_hide_manual(self, hide: bool):
        logging.info(f"[viewmodel] set_filter_hide_manual called: hide={hide}, current={self._filter_ctrl.hide_manual}")
        if self._filter_ctrl.set_hide_manual(hide):
            logging.info(f"[viewmodel] Filter changed, applying filters. hide_manual={self._filter_ctrl.hide_manual}")
            self._apply_filters()
            self._emit_state_snapshot()
        else:
            logging.info(f"[viewmodel] Filter unchanged, no action taken")

    def clear_filters(self):
        if self._filter_ctrl.clear():
            self._apply_filters()
            self._emit_state_snapshot()

    def current_filtered_images(self):
        return list(self._filtered_images)
    
    def _ensure_selection(self):
        """Ensure at least one image is always selected. Selects first filtered image if none selected."""
        # Guard against infinite recursion
        if getattr(self, "_ensuring_selection", False):
            return
        self._ensuring_selection = True
        try:
            if not self._filtered_images:
                # No images available, clear selection
                if self.selected_image:
                    self.selection_model.clear()
                    self.selected_image = None
                    self._has_selected_image = False
                    self.selected_image_changed.emit("")
                    self.has_selected_image_changed.emit(False)
                return
            
            # Check primary from selection_model first (it's the source of truth)
            primary = self.selection_model.primary()
            if primary:
                # Primary is set, use it
                if self.selected_image != primary:
                    self.selected_image = primary
                    self._has_selected_image = True
                    self.selected_image_changed.emit(primary)
                    self.has_selected_image_changed.emit(True)
                    filename = os.path.basename(primary)
                    for fname in self.images:
                        path = self._image_path_cache.get(fname) if hasattr(self, "_image_path_cache") else self.model.get_image_path(fname)
                        if path == primary:
                            filename = fname
                            break
                    self._load_details_for_filename(filename)
                return
            
            # If no image is selected, select the first filtered image
            if not self.selected_image or not self._has_selected_image:
                first_filename = self._filtered_images[0]
                first_path = self._image_path_cache.get(first_filename) if hasattr(self, "_image_path_cache") else self.model.get_image_path(first_filename)
                if first_path:
                    # Use selection_model directly to avoid recursion
                    self.selection_model.replace(first_path)
                    self.selected_image = first_path
                    self._has_selected_image = True
                    self.selected_image_changed.emit(first_path)
                    self.has_selected_image_changed.emit(True)
                    self._load_details_for_filename(first_filename)
            else:
                # Check if selected image is still in filtered list
                selected_filename = None
                for fname in self.images:
                    path = self._image_path_cache.get(fname) if hasattr(self, "_image_path_cache") else self.model.get_image_path(fname)
                    if path == self.selected_image:
                        selected_filename = fname
                        break
                
                if selected_filename and selected_filename not in self._filtered_images:
                    # Selected image is filtered out, select first filtered image
                    first_filename = self._filtered_images[0]
                    first_path = self._image_path_cache.get(first_filename) if hasattr(self, "_image_path_cache") else self.model.get_image_path(first_filename)
                    if first_path:
                        # Use selection_model directly to avoid recursion
                        self.selection_model.replace(first_path)
                        self.selected_image = first_path
                        self._has_selected_image = True
                        self.selected_image_changed.emit(first_path)
                        self.has_selected_image_changed.emit(True)
                        self._load_details_for_filename(first_filename)
        finally:
            self._ensuring_selection = False

    def active_filters(self) -> dict:
        return {
            "rating": self._filter_ctrl.rating if self._filter_ctrl.rating else 0,
            "tag": self._filter_ctrl.tag if self._filter_ctrl.tag else "",
            "date": self._filter_ctrl.date if self._filter_ctrl.date else "",
            "hide_manual": self._filter_ctrl.hide_manual,
        }

    def set_filters(self, rating=None, tag=None, date=None, hide_manual=None, emit=True):
        if self._filter_ctrl.set_batch(rating=rating, tag=tag, date=date, hide_manual=hide_manual):
            self._apply_filters()
            if emit:
                self._emit_state_snapshot()

    def load_images(self):
        logging.info(f"PhotoViewModel.load_images directory={self.model.directory}")
        self.thumbnail_service.cancel_all()
        self.exif_service.cancel_all()
        self._thumb_requested.clear()
        self._progress_current = 0
        self.images = []
        # Invalidate caches when images change
        if hasattr(self, "_image_path_cache"):
            delattr(self, "_image_path_cache")
        if hasattr(self, "_image_basename_cache"):
            delattr(self, "_image_basename_cache")
        files = self.model.get_image_files()
        self._progress_total = len(files)
        # Reset grouping when images change
        self._grouping_computed = False
        self._group_info = {}
        self.images_changed.emit(self.images)
        self.progress_changed.emit(self._progress_current, self._progress_total)
        # Defer state snapshot to event loop to avoid blocking startup
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._emit_state_snapshot)

        # Small batch or headless: sync load
        if self._progress_total <= 50 or self._is_headless():
            self._load_images_sync(files)
            return

        # GUI path: incremental batches
        self.task_started.emit("load-images")
        self._files_to_load = files
        self._load_index = 0
        self._process_next_batch()

    def _is_headless(self) -> bool:
        try:
            from PySide6.QtWidgets import QApplication

            return QApplication.instance() is None
        except Exception:
            return False

    def _load_images_sync(self, files):
        for idx, filename in enumerate(files):
            self.images.append(filename)
            self.image_added.emit(filename, idx)
            self._progress_current = idx + 1
            self.progress_changed.emit(self._progress_current, self._progress_total)

        self._apply_filters()
        # Compute grouping asynchronously (non-blocking)
        self._compute_grouping_async()
        self._emit_state_snapshot()
        if (self._auto and self._auto.enabled) or getattr(self, "_auto_label_enabled", False):
            # Removed previous red/green color fallback; rely on model predictions only
            self._auto.auto_label_unlabeled_async()
            self._auto.update_predictions_async()
            self._auto.ensure_initial_labels()

        self.progress_changed.emit(self._progress_total, self._progress_total)

    def _process_next_batch(self, batch_size=20):
        import time
        t0 = time.perf_counter()
        if self._load_index >= len(self._files_to_load):
            self._finalize_image_loading()
            return

        end_index = min(self._load_index + batch_size, len(self._files_to_load))
        for idx in range(self._load_index, end_index):
            filename = self._files_to_load[idx]
            self.images.append(filename)
            self.image_added.emit(filename, idx)
            self._progress_current = idx + 1
            self.progress_changed.emit(self._progress_current, self._progress_total)
            self.task_progress.emit("load-images", idx + 1, self._progress_total, filename)

        self._load_index = end_index
        # Re-apply filters to maintain sorting by confidence when images are added
        self._apply_filters()
        t1 = time.perf_counter()
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._process_next_batch)

    def _finalize_image_loading(self):
        self._apply_filters()
        # Compute grouping asynchronously (non-blocking)
        self._compute_grouping_async()
        # Ensure at least one image is selected after loading completes
        self._ensure_selection()
        
        # Auto-select image with most bounding boxes for testing (only once)
        if self._detected_objects and not self._auto_selected_most_boxes:
            max_count = 0
            max_file = None
            for filename, detections in self._detected_objects.items():
                count = len(detections) if detections else 0
                if count > max_count:
                    max_count = count
                    max_file = filename
            if max_file:
                logging.debug(f"[DEBUG] Auto-selecting image with most detections: {max_file} ({max_count} boxes)")
                self.select_image(max_file)
                self._auto_selected_most_boxes = True
        
        self._emit_state_snapshot()

        if getattr(self, "_auto_label_enabled", False) or (self._auto and self._auto.enabled):
            self._auto.auto_label_unlabeled_async()

        self._auto.update_predictions_async()

        self._progress_current = self._progress_total
        self.progress_changed.emit(self._progress_current, self._progress_total)
        self.task_finished.emit("load-images", True)

    def set_label(self, label: str):
        self._label_ctrl.set_label(label)

    def _update_rating_tags(self):
        self._label_ctrl.update_rating_tags()

    def select_image(self, filename: str):
        path = self.model.get_image_path(filename)
        if not path:
            return

        self.selection_model.replace(path)
        primary_changed = self.selected_image != path
        self.selected_image = path
        self._has_selected_image = True

        if primary_changed:
            self.selected_image_changed.emit(path)
            self.has_selected_image_changed.emit(True)

        self._load_details_for_filename(os.path.basename(path))

        # Always apply filters to maintain sorting by uncertainty, even when no filters are active
        self._apply_filters()
        self._emit_state_snapshot()

    def handle_selection_click(self, filename: str, modifiers):
        path = self.model.get_image_path(filename)
        if not path:
            return

        from PySide6.QtCore import Qt

        ctrl = bool((modifiers & Qt.KeyboardModifier.ControlModifier) or (modifiers & Qt.KeyboardModifier.MetaModifier))
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if shift:
            ordered = [self.model.get_image_path(f) for f in self.images if self.model.get_image_path(f)]
            self.selection_model.extend_range(path, ordered)
        elif ctrl:
            self.selection_model.toggle(path)
        else:
            self.selection_model.replace(path)

    def _on_selection_changed(self, selected_paths):
        had = self._has_selected_image
        self._has_selected_image = bool(selected_paths)

        if self._has_selected_image != had:
            self.has_selected_image_changed.emit(self._has_selected_image)

        # Only clear selected_image if selection is empty AND primary hasn't been set yet
        # The primaryChanged signal will update selected_image, so we don't want to clear it here
        # unless the selection is actually empty
        if not selected_paths and self.selected_image:
            self.selected_image = None
        elif selected_paths and self.selected_image not in selected_paths:
            # Selection changed but current selected_image is not in new selection
            # Wait for primaryChanged to update it, don't clear it here
            pass

        self.selection_changed.emit(selected_paths)
        
        # Ensure at least one image is selected after selection changes
        # Only if we're not already ensuring selection (to avoid recursion)
        if not getattr(self, "_ensuring_selection", False):
            self._ensure_selection()
        
        self._emit_state_snapshot()

    def _on_primary_changed(self, primary_path):
        if primary_path:
            self.selected_image = primary_path
            self.selected_image_changed.emit(primary_path)
            self._request_exif_for_selected()
        else:
            self.selected_image = None
            self.exif = {}
            self.exif_changed.emit(self.exif)
            # Ensure at least one image is selected after primary is cleared
            self._ensure_selection()

        self._emit_state_snapshot()

    def _request_exif_for_selected(self):
        if not self.selected_image or not self.exif_service:
            return

        path = self.selected_image

        if path in self._exif_cache:
            self.exif = self._exif_cache.get(path, {})
            self.exif_changed.emit(self.exif)
            return

        if path in self._exif_inflight:
            return

        self._exif_inflight.add(path)

        def callback(cb_path: str, data: dict):
            self._exif_inflight.discard(cb_path)
            self._exif_cache[cb_path] = data or {}
            if self.selected_image == cb_path:
                self.exif = self._exif_cache.get(cb_path, {})
                self.exif_changed.emit(self.exif)

        self.exif_service.request_exif(path, callback)

    def cleanup(self):
        if getattr(self, "_cleaned", False):
            return

        self._cleaned = True
        logging.info("PhotoViewModel cleanup starting...")
        
        # Cancel all thumbnail and EXIF requests
        try:
            self.thumbnail_service.cancel_all()
            self.exif_service.cancel_all()
        except Exception as e:
            logging.warning(f"Error cancelling services: {e}")
        
        # Stop detection worker
        try:
            stop_global_worker()
        except Exception as e:
            logging.warning(f"Error stopping detection worker: {e}")
        
        # Stop all TaskRunner threads
        try:
            if hasattr(self, "_tasks") and self._tasks:
                self._tasks.shutdown(timeout_ms=2000)
        except Exception as e:
            logging.warning(f"Error stopping TaskRunner: {e}")
        
        # Terminate multiprocessing pools
        try:
            from src.features import _cleanup_pools
            _cleanup_pools()
        except Exception as e:
            logging.warning(f"Error cleaning up feature extraction pools: {e}")
        
        logging.info("PhotoViewModel cleanup complete")

    def open_selected_in_viewer(self):
        if not self.selected_image:
            logging.warning("No image selected to open in viewer.")
            return

        import subprocess
        import sys

        path = self.selected_image

        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            logging.error(f"Failed to open image in viewer: {e}")

    def _emit_state_snapshot(self):
        """Debounced state snapshot emitter - schedules actual work via timer."""
        self._snapshot_pending = True
        if not self._snapshot_timer.isActive():
            self._snapshot_timer.start(self._snapshot_debounce_ms)

    def _emit_state_snapshot_immediate(self):
        """Actually emit the state snapshot (called by debounce timer)."""
        if not self._snapshot_pending:
            return
        self._snapshot_pending = False
        
        # Load object detections for current images (fast cache-only)
        try:
            self._load_object_detections()
        except Exception as e:
            logging.warning(f"Failed to load object detections: {e}")
            # Continue even if detections fail
        
        # Re-apply filters to ensure sorting by uncertainty is up-to-date
        try:
            self._apply_filters()
        except Exception as e:
            logging.warning(f"Failed to apply filters: {e}")
            # Continue even if filters fail

        # Try to detect which detection backend/device/model is in use for UI display
        detection_backend = "unknown"
        detection_device = "unknown"
        detection_model = "unknown"
        try:
            from . import object_detection as _od

            detection_backend = getattr(_od, "DETECTION_BACKEND", detection_backend)
            detection_device = _od.get_loaded_device() or detection_device
            detection_model = _od.get_loaded_model_name() or detection_model
        except Exception:
            # ignore if object_detection can't be imported
            pass

        # Get classification model info
        classification_model_type = "unknown"
        classification_model_path = "unknown"
        try:
            from .inference import load_model
            from .training_core import DEFAULT_MODEL_PATH
            import os
            
            model_path = os.path.expanduser(DEFAULT_MODEL_PATH)
            if os.path.isfile(model_path):
                bundle = load_model(model_path)
                if bundle:
                    # Detect model type (XGBoost or CatBoost)
                    model = bundle.model
                    if hasattr(model, "named_steps"):
                        if "cat" in model.named_steps:
                            classification_model_type = "CatBoost"
                        elif "xgb" in model.named_steps:
                            classification_model_type = "XGBoost"
                    classification_model_path = os.path.basename(model_path)
        except Exception:
            pass

        # Get embedding model info (ResNet18 with device detection)
        embedding_model = "ResNet18"
        embedding_device = "unknown"
        try:
            from .inference import _get_embedding_device_used
            embedding_device = _get_embedding_device_used().upper()
        except Exception:
            pass

        # Get feature extraction backend info
        feature_extraction_backend = "PIL/OpenCV/NumPy"

        snapshot = ImageBrowserState(
            images=list(self.images),
            selected=self.selection_model.selected(),
            primary=self.selection_model.primary(),
            rating=self._rating,
            tags=list(self._tags),
            label=self._label,
            has_selection=bool(self.selection_model.selected()),
            progress_current=self._progress_current,
            progress_total=self._progress_total,
            can_undo=self.can_undo,
            can_redo=self.can_redo,
            filtered_images=list(self._filtered_images),
            filtered_count=len(self._filtered_images),
            predicted_labels=dict(self._auto.predicted_labels),
            predicted_probabilities=dict(self._auto.predicted_probabilities),
            auto_assigned_paths=set(self._auto.auto_assigned),
            detected_objects=dict(self._detected_objects),
            detection_backend=detection_backend,
            detection_device=str(detection_device),
            detection_model=detection_model,
            classification_model_type=classification_model_type,
            classification_model_path=classification_model_path,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            feature_extraction_backend=feature_extraction_backend,
            group_info=self._group_info.copy(),  # Include grouping data
        )
        try:
            self.browser_state_changed.emit(snapshot)
        except RuntimeError:
            # The UI receiver was likely deleted during shutdown; ignore
            logging.debug("PhotoViewModel: browser_state_changed signal ignored, receiver deleted")
        except Exception as e:
            logging.error(f"Failed to emit browser_state_changed: {e}")
            raise

    def _filters_active(self) -> bool:
        result = self._filter_ctrl.active()
        return bool(result)

    def execute_command(self, cmd):
        self._cmd_stack.execute(cmd)
        self._emit_undo_redo()

    def undo(self):
        if self._cmd_stack.can_undo:
            self._cmd_stack.undo()
            if self.selected_image:
                self._load_details_for_filename(os.path.basename(self.selected_image))
            if self._filters_active():
                self._apply_filters()
            self._emit_state_snapshot()
            self._emit_undo_redo()

    def redo(self):
        if self._cmd_stack.can_redo:
            self._cmd_stack.redo()
            if self.selected_image:
                self._load_details_for_filename(os.path.basename(self.selected_image))
            if self._filters_active():
                self._apply_filters()
            self._emit_state_snapshot()
            self._emit_undo_redo()

    def load_thumbnail(self, filename):
        path = self.model.get_image_path(filename)
        if not path or path in self._thumb_requested:
            return

        self._thumb_requested.add(path)
        self.thumbnail_service.request_thumbnail(path, 128, self._on_thumbnail_result)

    def _on_thumbnail_result(self, path, image):
        logging.debug(f"[viewmodel] _on_thumbnail_result for {path}, has_image={image is not None}")
        self.thumbnail_loaded.emit(path, image)

    def _load_details_for_filename(self, filename):
        path = self.model.get_image_path(filename)
        if not path:
            return
        self._rating = self.model.get_rating(path)
        self._tags = self.model.get_tags(path)
        self._label = self.model.get_state(path) or None
        self.rating_changed.emit(self._rating)
        self.tags_changed.emit(self._tags)
        if self._label:
            self.label_changed.emit(filename, self._label)

    def _apply_pending_label_changes(self):
        pending = dict(self._pending_label_changes)
        self._pending_label_changes.clear()

        if not pending:
            return

        logging.info(f"[label-queue] Applying {len(pending)} pending manual label changes")

        for path, label in pending.items():
            self._cmd_stack.execute_or_direct(SetLabelCommand, self.model, path, label)

            if self.selected_image == path:
                self._label = label

            if path in self._auto.auto_assigned:
                self._auto.auto_assigned.discard(path)

            filename = os.path.basename(path)
            self.label_changed.emit(filename, label)
            # Notify AutoLabelManager of label change (for debounced re-prediction)
            self._auto.notify_label_changed(filename, label)

        will_retrain = any(label in ("keep", "trash") for label in pending.values())
        if will_retrain:
            self._auto.schedule_retrain()
            # Skip immediate prediction update - post-retrain workflow will handle it
        else:
            # For non-keep/trash changes, update predictions immediately
            self._auto.update_predictions_async()

        # Always apply filters to maintain sorting by uncertainty, even when no filters are active
        # Note: _on_label_changed will also call _apply_filters(), but we call it here too
        # to ensure sorting happens even if signal connection fails
        self._apply_filters()
        self._emit_state_snapshot()
    
    def _on_prediction_updated(self, filename: str, probability: float, label: str):
        """Handle prediction update - re-sort images by uncertainty (debounced)."""
        # PERFORMANCE: Debounce _apply_filters to avoid excessive calls
        self._apply_filters_pending = True
        if not self._apply_filters_timer.isActive():
            self._apply_filters_timer.timeout.connect(self._apply_filters_debounced)
            self._apply_filters_timer.start(100)  # 100ms debounce (10 calls/sec max)
        # Emit state snapshot to update UI with new sort order
        self._emit_state_snapshot()
    
    def _apply_filters_debounced(self):
        """Debounced version of _apply_filters."""
        self._apply_filters_timer.timeout.disconnect()
        if self._apply_filters_pending:
            self._apply_filters_pending = False
            self._apply_filters()
    
    def _on_label_changed(self, filename: str, label: str):
        """Handle label change - re-sort images by uncertainty (debounced)."""
        # PERFORMANCE: Debounce _apply_filters to avoid excessive calls
        self._apply_filters_pending = True
        if not self._apply_filters_timer.isActive():
            self._apply_filters_timer.timeout.connect(self._apply_filters_debounced)
            self._apply_filters_timer.start(100)  # 100ms debounce (10 calls/sec max)
        # Emit state snapshot to update UI with new sort order
        self._emit_state_snapshot()

    # ---------------- Manual Training (non-blocking) -----------------
    def start_training(self):
        """Begin no-blocking training in background thread pool.
        Safe no-op if already active."""
        if self._training_active:
            logging.info("[ui-train] Training already active; ignoring start request")
            return
        repo = getattr(self.model, "_repo", None)
        directory = self.model.directory
        if not directory or not os.path.isdir(directory):
            logging.warning("[ui-train] Invalid directory for training: %s", directory)
            return

        self._training_cancel_requested = False
        self._training_active = True

        def _task(reporter):
            from .training_core import DEFAULT_MODEL_PATH, train_keep_trash_model

            try:

                def progress_cb(current, total, detail):
                    try:
                        reporter.update(current, total)
                        reporter.detail(detail)
                    except Exception:
                        logging.exception("Error in viewmodel background task")
                        raise

                def cancellation_token():
                    return self._training_cancel_requested

                reporter.detail("starting dataset build")
                result = train_keep_trash_model(
                    image_dir=directory,
                    model_path=DEFAULT_MODEL_PATH,
                    repo=repo,
                    n_estimators=200,
                    progress_callback=progress_cb,
                    cancellation_token=cancellation_token,
                )
                if self._training_cancel_requested:
                    reporter.detail("cancelled")
                    logging.info("[ui-train] Training cancelled by user")
                    return
                if result:
                    # Compute thresholds via auto manager if present
                    keep_eff, trash_eff = self._auto.compute_dynamic_thresholds()
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
                        "confusion": result.confusion,
                    }
                    # Add model metadata for feature name mapping (interactions, embeddings)
                    try:
                        from .inference import load_model
                        if result.model_path:
                            bundle = load_model(result.model_path)
                            if bundle and bundle.meta:
                                stats_dict["model_metadata"] = bundle.meta
                    except Exception:
                        pass  # Ignore errors loading metadata
                    if keep_eff is not None and trash_eff is not None and hasattr(self, "_auto"):
                        stats_dict.update(
                            {
                                "keep_threshold_base": self._auto.keep_thr,
                                "trash_threshold_base": self._auto.trash_thr,
                                "keep_threshold_eff": keep_eff,
                                "trash_threshold_eff": trash_eff,
                                "weighted_mode": self._auto.weighted_mode,
                            }
                        )
                    self.model_stats_changed.emit(stats_dict)
                    reporter.detail("done")
                else:
                    reporter.detail("skipped or cancelled")
            finally:
                self._training_active = False

        self._tasks.run("training", _task)

    def cancel_training(self):
        """Request cancellation of an in-progress training task."""
        if not self._training_active:
            logging.info("[ui-train] No active training to cancel")
            return
        self._training_cancel_requested = True
        logging.info("[ui-train] Cancellation requested")

    def is_training_active(self) -> bool:
        return self._training_active

    def set_rating(self, rating: int):
        """Backward-compatible pass-through to LabelController.set_rating."""
        self._label_ctrl.set_rating(int(rating))

    def set_tags(self, tags: list):
        """Backward-compatible pass-through to LabelController.set_tags."""
        self._label_ctrl.set_tags(list(tags))

    def _refresh_auto_labels(self):
        """Backward-compatible wrapper calling AutoLabelManager.refresh_auto_labels."""
        self._auto.refresh_auto_labels()
    
    def _on_grouping_completed(self):
        """Handle grouping completion - emit state snapshot from main thread."""
        logging.info(f"[grouping] Signal received, emitting state snapshot (group_info has {len(self._group_info)} entries)")
        try:
            # Re-apply filters to re-sort with new group info
            self._apply_filters()
            self._emit_state_snapshot()
        except Exception as e:
            logging.exception(f"[grouping] Failed to emit state snapshot: {e}")
    
    def _compute_grouping_async(self):
        """Compute photo grouping asynchronously (non-blocking)."""
        if self._grouping_computed:
            logging.debug(f"[grouping] Already computed, skipping (images={len(self.images)})")
            return
        if not self.images:
            logging.debug("[grouping] No images to group")
            return
        
        # Mark as in-progress to avoid duplicate computation
        self._grouping_computed = True
        logging.info(f"[grouping] Starting async computation for {len(self.images)} photos")
        
        def compute_grouping(reporter=None):
            try:
                if reporter:
                    reporter.set_total(8)  # 8 steps total
                    reporter.detail("Starting grouping computation...")
                
                logging.info(f"[grouping] Computing grouping for {len(self.images)} photos")
                
                # Collect EXIF data
                if reporter:
                    reporter.update(1, 8)
                    reporter.detail("Collecting EXIF data...")
                exif_data: dict[str, dict] = {}
                for idx, filename in enumerate(self.images):
                    path = self.model.get_image_path(filename)
                    if path:
                        exif_data[filename] = self.model.load_exif(path)
                    if reporter and (idx + 1) % 100 == 0:
                        reporter.detail(f"Collected EXIF for {idx + 1}/{len(self.images)} photos...")
                
                # Get keep probabilities from auto-label manager
                keep_probs: dict[str, float] = {}
                if self._auto:
                    keep_probs = {
                        filename: self._auto.predicted_probabilities.get(filename, 0.5)
                        for filename in self.images
                    }
                
                # Compute grouping with progress reporting
                if reporter:
                    reporter.update(2, 8)
                    reporter.detail("Computing photo groups...")
                
                group_info = compute_grouping_for_photos(
                    filenames=self.images,
                    image_dir=self.model.directory,
                    exif_data=exif_data,
                    keep_probabilities=keep_probs if keep_probs else None,
                    quality_metrics=None,  # TODO: extract quality metrics from features if available
                    progress_reporter=reporter,  # Pass reporter for progress updates
                    # phash_threshold uses default from PHASH_HAMMING_THRESHOLD constant (8)
                )
                
                # Update group_info
                self._group_info = group_info
                group_count = len(set(g.get('group_id', 0) for g in group_info.values() if g.get('group_id') is not None))
                best_count = sum(1 for g in group_info.values() if g.get('is_group_best', False))
                logging.info(f"[grouping] Computed grouping: {group_count} groups, {best_count} best picks")
                
                # Emit signal to update UI (signals are thread-safe in Qt)
                try:
                    self.grouping_completed.emit()
                    logging.debug("[grouping] Signal emitted")
                except RuntimeError as e:
                    # Object deleted or signal not connected, log but don't fail
                    logging.debug(f"[grouping] Signal emit failed (likely app closing): {e}")
                except Exception as e:
                    logging.warning(f"[grouping] Unexpected error emitting signal: {e}")
            except Exception as e:
                logging.exception(f"[grouping] Failed to compute grouping: {e}")
                self._grouping_computed = False  # Allow retry on error
        
        # Run in background thread
        if hasattr(self, "_tasks") and self._tasks:
            try:
                self._tasks.run("grouping", compute_grouping)
            except Exception as e:
                logging.warning(f"[grouping] Failed to submit task, using fallback: {e}")
                # Fallback: run in background thread (without progress reporting)
                import threading
                thread = threading.Thread(target=lambda: compute_grouping(None), daemon=True)
                thread.start()
        else:
            # Fallback: run in background thread if task runner not available
            import threading
            thread = threading.Thread(target=lambda: compute_grouping(None), daemon=True)
            thread.start()
