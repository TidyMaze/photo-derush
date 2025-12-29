from __future__ import annotations

"""Directory management for PhotoViewModel (SRP)."""
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .viewmodel import PhotoViewModel
from .model import RatingsTagsRepository


def change_directory(vm: PhotoViewModel, new_directory: str) -> bool:
    if not os.path.isdir(new_directory):
        logging.error(f"Directory does not exist: {new_directory}")
        return False
    try:
        repo_path = os.path.join(new_directory, ".ratings_tags.json")
        new_repo = RatingsTagsRepository(path=repo_path)
        new_repo._data = None  # force fresh load
        vm.model.directory = new_directory
        vm.model._repo = new_repo
        # Reset state
        vm.selected_image = None
        vm._has_selected_image = False
        vm.exif = {}
        vm._rating = 0
        vm._tags = []
        vm._label = None
        vm.images = []
        vm._filtered_images = []
        vm._thumb_requested.clear()
        vm._exif_cache.clear()
        vm._exif_inflight.clear()
        vm.selection_model.clear()
        # Reset ML prediction maps
        vm._auto.predicted_labels.clear()
        vm._auto.predicted_probabilities.clear()
        vm._auto.auto_assigned.clear()
        logging.info(f"[vm_directory] Changed directory to: {new_directory}")
        vm.directory_changed.emit(new_directory)
        vm.load_images()
        return True
    except Exception as e:
        logging.error(f"Failed to change directory: {e}")
        return False


__all__ = ["change_directory"]
