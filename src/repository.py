"""Repository module: persistence for ratings, tags, and label state.

Single responsibility: manage loading/saving JSON store keyed by filename.
Separated from model.py to reduce coupling with image/exif logic.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

# Default to current directory's .ratings_tags.json
RATINGS_TAGS_PATH = ".ratings_tags.json"


class RatingsTagsRepository:
    """Persistence layer for ratings / tags / state (keep|trash).
    Stores data keyed by filename (no directories) to keep portability.
    Stores .ratings_tags.json in the dataset directory.
    Responsible ONLY for persistence + basic CRUD; validation handled by callers.
    """

    def __init__(self, path: str | None = None):
        # Use provided path, or default to current directory's .ratings_tags.json
        self.path = path if path else RATINGS_TAGS_PATH
        self._data: dict[str, dict[str, Any]] | None = None

    # ---------------- Internal helpers -----------------
    def _ensure_loaded(self):
        if self._data is not None:
            return
        changed = False
        try:
            with open(self.path) as f:
                raw = json.load(f)
            cleaned: dict[str, dict[str, Any]] = {}
            for k, v in raw.items():
                filename = os.path.basename(k)
                if filename in cleaned:
                    merged = cleaned[filename].copy()
                    merged.update(v)
                    cleaned[filename] = merged
                else:
                    cleaned[filename] = v
                if k != filename:
                    changed = True
            self._data = cleaned
            if changed:
                self._save()  # persist migration
            logging.debug("RatingsTagsRepository loaded %d entries", len(self._data))
        except Exception as e:
            logging.debug("RatingsTagsRepository starting new store: %s", e)
            self._data = {}

    def _save(self):
        if self._data is None:
            return
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(self.path)
            if parent_dir:
                parent_dir = os.path.abspath(parent_dir)
                if not os.path.exists(parent_dir):
                    try:
                        os.makedirs(parent_dir, exist_ok=True)
                        logging.debug(f"[repository] Created parent directory: {parent_dir}")
                    except OSError as dir_err:
                        logging.error(f"[repository] Failed to create directory {parent_dir}: {dir_err}")
                        raise

            # Write directly to target file (simpler, more robust than temp file + rename)
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
            logging.debug(f"[repository] Saved {len(self._data)} ratings/tags to {self.path}")
        except Exception as e:
            logging.warning("Failed saving ratings/tags: %s", e)

    # ---------------- Public CRUD API -----------------
    def get(self, filename: str) -> dict[str, Any]:
        self._ensure_loaded()
        return self._data.get(filename, {})  # type: ignore[arg-type]

    def set_rating(self, filename: str, rating: int):
        self._ensure_loaded()
        assert self._data is not None
        entry = self._data.setdefault(filename, {})
        entry["rating"] = int(rating)
        self._save()

    def get_rating(self, filename: str) -> int:
        return int(self.get(filename).get("rating", 0))

    def set_tags(self, filename: str, tags):
        self._ensure_loaded()
        assert self._data is not None
        entry = self._data.setdefault(filename, {})
        entry["tags"] = tags
        self._save()

    def get_tags(self, filename: str):
        return list(self.get(filename).get("tags", []))

    def set_state(self, filename: str, state: str, source: str = "manual"):
        """Set the label state ('keep'|'trash' or '' to clear) and its source.
        Protection: manual labels are immutable to 'auto' updates.
        """
        self._ensure_loaded()
        assert self._data is not None
        if state not in ("keep", "trash", ""):
            logging.error("Invalid state: %s", state)
            return

        entry = self._data.get(filename, {})
        existing_state = entry.get("state")
        existing_source = entry.get("label_source", "manual")

        # Helper: check if trying to auto-overwrite manual label
        is_auto_overwrite = existing_state and existing_source == "manual" and source != "manual"

        # Block auto attempting to overwrite manual
        if is_auto_overwrite:
            logging.debug(f"[repository] Skip auto overwrite of manual label: {filename} {existing_state}->{state}")
            return

        # Block auto clearing a manual label
        if state == "" and is_auto_overwrite:
            logging.debug(f"[repository] Skip auto clear of manual label: {filename}")
            return

        if state:
            entry = self._data.setdefault(filename, {})
            entry["state"] = state
            entry["label_source"] = source
        else:
            if filename in self._data:
                entry = self._data[filename]
                entry.pop("state", None)
                # Preserve label_source only if other metadata remains; else delete entry
                if not entry:
                    del self._data[filename]
        self._save()

    def get_state(self, filename: str) -> str:
        result = self.get(filename).get("state", "")
        return str(result) if result else ""

    def set_objects(self, filename: str, objects):
        self._ensure_loaded()
        assert self._data is not None
        entry = self._data.setdefault(filename, {})
        entry["objects"] = objects
        self._save()

    def get_objects(self, filename: str):
        return list(self.get(filename).get("objects", []))

    def get_label_source(self, filename: str) -> str:
        result = self.get(filename).get("label_source", "manual")
        return str(result) if result else "manual"
