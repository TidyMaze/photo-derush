from PySide6.QtCore import QObject, Signal


class SelectionModel(QObject):
    selectionChanged = Signal(list)  # list of selected full paths
    primaryChanged = Signal(object)  # primary full path or None

    def __init__(self):
        super().__init__()
        self._selected: set[str] = set()
        self._primary: str | None = None
        self._anchor: str | None = None  # for shift range operations

    def clear(self):
        changed = bool(self._selected) or self._primary is not None
        self._selected.clear()
        self._primary = None
        self._anchor = None
        if changed:
            self.selectionChanged.emit([])
            self.primaryChanged.emit(None)

    def set(self, paths: list[str], primary: str | None = None):
        paths_set = {p for p in paths if p}
        self._selected = paths_set
        if primary and primary in paths_set:
            self._primary = primary
        else:
            self._primary = next(iter(paths_set)) if paths_set else None
        self._anchor = self._primary
        self.selectionChanged.emit(list(self._selected))
        self.primaryChanged.emit(self._primary)

    def toggle(self, path: str):
        if not path:
            return
        if path in self._selected:
            self._selected.remove(path)
            if self._primary == path:
                self._primary = next(iter(self._selected)) if self._selected else None
        else:
            self._selected.add(path)
            self._primary = path  # make toggled item primary
            self._anchor = path
        self.selectionChanged.emit(list(self._selected))
        self.primaryChanged.emit(self._primary)

    def replace(self, path: str):
        if not path:
            return
        self._selected = {path}
        self._primary = path
        self._anchor = path
        self.selectionChanged.emit([path])
        self.primaryChanged.emit(path)

    def extend_range(self, path: str, ordered_paths: list[str]):
        if not path or not ordered_paths:
            return
        anchor_valid = self._anchor is not None and self._anchor in ordered_paths
        if not anchor_valid:
            self._anchor = path
        try:
            start_index = ordered_paths.index(self._anchor)
            end_index = ordered_paths.index(path)
        except ValueError:
            # fallback to replace if path not found
            self.replace(path)
            return
        lo, hi = sorted((start_index, end_index))
        rng = ordered_paths[lo : hi + 1]
        self._selected.update(rng)
        self._primary = path
        self.selectionChanged.emit(list(self._selected))
        self.primaryChanged.emit(self._primary)

    def selected(self) -> list[str]:
        return list(self._selected)

    def primary(self) -> str | None:
        return self._primary

    def is_selected(self, path: str) -> bool:
        return path in self._selected
