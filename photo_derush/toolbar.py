from PySide6.QtWidgets import QToolBar, QComboBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal

class SettingsToolbar(QToolBar):
    zoom_changed = Signal(int)
    def __init__(self, parent=None):
        super().__init__("Settings Toolbar", parent)
        self.sort_by_group_action = QAction("Sort by group", self)
        self.sort_by_group_action.setCheckable(True)
        self.addAction(self.sort_by_group_action)
        # Zoom selector
        self.zoom_selector = QComboBox(self)
        self.zoom_selector.addItems(["80", "120", "160", "200", "240"])
        self.zoom_selector.setCurrentText("160")
        self.zoom_selector.setToolTip("Zoom (cell size)")
        self.addWidget(self.zoom_selector)
        self.zoom_selector.currentTextChanged.connect(self._on_zoom_changed)
    def _on_zoom_changed(self, value):
        self.zoom_changed.emit(int(value))
