"""Interface for brain module to interact with data layer without UI dependencies.

This allows the brain module (training, classification, auto-labeling) to be used
independently of the UI, enabling future use cases like Lightroom plugins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class ImageDataProvider(Protocol):
    """Protocol for accessing image data without UI dependencies."""

    def get_image_path(self, filename: str) -> str | None:
        """Get full path for an image filename."""
        ...

    def get_state(self, path: str) -> str:
        """Get label state ('keep', 'trash', or '')."""
        ...

    def set_state(self, path: str, state: str, source: str = "manual") -> None:
        """Set label state."""
        ...

    def get_image_files(self) -> list[str]:
        """Get list of image filenames in directory."""
        ...

    def get_label_source(self, filename: str) -> str:
        """Get label source ('manual' or 'auto')."""
        ...

    @property
    def directory(self) -> str:
        """Get image directory path."""
        ...

    @property
    def _repo(self):
        """Get repository instance (for label source queries)."""
        ...


class BrainCallbacks:
    """Optional callbacks for UI integration (can be None for headless use)."""

    def __init__(
        self,
        on_label_changed: Callable[[str, str], None] | None = None,
        on_prediction_updated: Callable[[str, float, str], None] | None = None,
        on_state_snapshot: Callable[[], None] | None = None,
        on_model_stats_changed: Callable[[dict], None] | None = None,
    ):
        self.on_label_changed = on_label_changed
        self.on_prediction_updated = on_prediction_updated
        self.on_state_snapshot = on_state_snapshot
        self.on_model_stats_changed = on_model_stats_changed


class TaskRunner(Protocol):
    """Protocol for running background tasks (optional, can be None for synchronous execution)."""

    def run(self, task_name: str, task_func: Callable) -> None:
        """Run a task in background."""
        ...


class BrainAdapter:
    """Adapter that wraps ImageModel to provide ImageDataProvider interface."""

    def __init__(self, model):
        self.model = model

    def get_image_path(self, filename: str) -> str | None:
        result = self.model.get_image_path(filename)
        return result if result is not None else None  # type: ignore[no-any-return]

    def get_state(self, path: str) -> str:
        return self.model.get_state(path)  # type: ignore[no-any-return]

    def set_state(self, path: str, state: str, source: str = "manual") -> None:
        self.model.set_state(path, state, source)

    def get_image_files(self) -> list[str]:
        return self.model.get_image_files()  # type: ignore[no-any-return]

    def get_label_source(self, filename: str) -> str:
        if hasattr(self.model, "_repo") and self.model._repo:
            return self.model._repo.get_label_source(filename)  # type: ignore[no-any-return]
        return "manual"

    @property
    def directory(self) -> str:
        return self.model.directory  # type: ignore[no-any-return]

    @property
    def _repo(self):
        return getattr(self.model, "_repo", None)

