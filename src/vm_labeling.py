"""Labeling and rating/tag operations (SRP)."""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from .commands import SetLabelCommand, SetRatingCommand, SetTagsCommand  # type: ignore

if TYPE_CHECKING:
    from .viewmodel import PhotoViewModel


class LabelController:
    """Handles labeling, rating, and tagging of photos in the view model."""

    def __init__(self, vm: PhotoViewModel):
        self.vm = vm
        self._retrain_lock = vm._retrain_lock if hasattr(vm, "_retrain_lock") else threading.Lock()

    def set_label(self, label: str):
        """Set label on selected image; schedules retrain if keep/trash changed."""
        logging.info(f"[label] set_label called with label='{label}'")
        vm = self.vm
        if not vm.selected_image:
            logging.warning(f"[label] set_label called but no image selected")
            return
        path = vm.selected_image
        filename = os.path.basename(path)
        current_label = vm.model.get_state(path)
        current_source = "manual"
        repo = getattr(vm.model, "_repo", None)
        if repo:
            try:
                current_source = repo.get_label_source(filename)
            except Exception:
                pass  # Default to manual if source fetch fails

        if vm._is_blocking_task_running():
            blocking_tasks = vm._active_tasks & {"retrain", "predict", "auto-label"}
            task_names = ", ".join(sorted(blocking_tasks))
            if current_label and current_source == "manual":
                # Queue manual change instead of blocking permanently
                if path not in vm._pending_label_changes or vm._pending_label_changes.get(path) != label:
                    vm._pending_label_changes[path] = label
                    logging.warning(f"[label] Queued manual change during {task_names} '{current_label}'→'{label}'")
                return

        label_changed = current_label != label
        source_changed = current_source == "auto" and label in ("keep", "trash")
        made_change = label_changed or source_changed
        logging.info(f"[label] set_label: filename={filename}, current_label={current_label}, new_label={label}, label_changed={label_changed}, source_changed={source_changed}, made_change={made_change}")
        if not made_change and current_label == label and current_source == "manual":
            return
        try:
            vm._cmd_stack.execute_or_direct(SetLabelCommand, vm.model, path, label)
        except Exception as e:
            raise

        # Mark as protected manual label and remove from auto tracking
        vm._auto.protected_manual_labels.add(filename)
        if path in vm._auto.auto_assigned:
            vm._auto.auto_assigned.discard(path)
        vm._auto.predicted_labels.pop(filename, None)

        vm._label = label
        try:
            vm.label_changed.emit(filename, label)
        except Exception as e:
            raise
        try:
            vm._emit_state_snapshot()
        except Exception as e:
            raise

        # Notify AutoLabelManager of label change (for debounced re-prediction)
        try:
            vm._auto.notify_label_changed(filename, label)
        except Exception as e:
            raise

        # Only retrain for keep/trash changes
        will_retrain = made_change and label in ("keep", "trash")
        if will_retrain:
            logging.info(f"[label] Label changed to '{label}', will_retrain={will_retrain}, made_change={made_change}")
            vm._nan_retrain_attempts = 0
            should_schedule = False
            with self._retrain_lock:
                if vm._retrain_in_progress:
                    vm._retrain_pending = True
                    logging.info("[label] Retrain in progress, queued")
                else:
                    vm._last_retrain_time = 0.0
                    should_schedule = True
            if should_schedule:
                try:
                    logging.info("[label] Calling schedule_retrain()")
                    vm._auto.schedule_retrain()
                except Exception as e:
                    logging.exception(f"[label] Failed to schedule retrain: {e}")
                    raise
        else:
            logging.debug(f"[label] Label changed to '{label}', will_retrain=False (not keep/trash or no change)")
            try:
                vm._auto.update_predictions_async()
            except Exception as e:
                logging.exception(f"[label] Failed to update predictions: {e}")
                raise

    def set_rating(self, rating: int):
        """Set rating on selected image (skip if unchanged)."""
        vm = self.vm
        if not vm.selected_image or rating == vm._rating:
            return
        path = vm.selected_image
        vm._cmd_stack.execute_or_direct(SetRatingCommand, vm.model, path, rating)
        vm._rating = rating
        vm.rating_changed.emit(vm._rating)
        if vm._filters_active():
            vm._apply_filters()
        vm._emit_state_snapshot()

    def set_tags(self, tags: list):
        """Set tag list on selected image; skips if identical to current."""
        vm = self.vm
        if not vm.selected_image:
            return
        path = vm.selected_image
        norm = list(tags or [])
        if norm == vm._tags:
            return
        vm._cmd_stack.execute_or_direct(SetTagsCommand, vm.model, path, norm)
        vm._tags = norm
        vm.tags_changed.emit(vm._tags)
        if vm._filters_active():
            vm._apply_filters()
        vm._emit_state_snapshot()

    def update_rating_tags(self):
        """Refresh in-memory rating/tags for current selection and emit changes."""
        vm = self.vm
        if vm.selected_image:
            vm._rating = vm.model.get_rating(vm.selected_image)
            vm._tags = vm.model.get_tags(vm.selected_image)
        else:
            vm._rating = 0
            vm._tags = []
        vm.rating_changed.emit(vm._rating)
        vm.tags_changed.emit(vm._tags)


__all__ = ["LabelController"]
