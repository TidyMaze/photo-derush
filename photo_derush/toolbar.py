from PySide6.QtWidgets import QToolBar, QComboBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Signal

class SettingsToolbar(QToolBar):
    zoom_changed = Signal(int)
    keep_clicked = Signal()
    trash_clicked = Signal()
    unsure_clicked = Signal()
    predict_sort_clicked = Signal()
    export_csv_clicked = Signal()
    reset_model_clicked = Signal()
    def __init__(self, parent=None):
        super().__init__("Settings Toolbar", parent)
        self.sort_by_group_action = QAction("Sort by group", self)
        self.sort_by_group_action.setCheckable(True)
        self.addAction(self.sort_by_group_action)
        # Keep/Trash/Unsure actions
        self.keep_action = QAction("Keep", self)
        self.trash_action = QAction("Trash", self)
        self.unsure_action = QAction("Unsure", self)
        self.addAction(self.keep_action)
        self.addAction(self.trash_action)
        self.addAction(self.unsure_action)
        self.keep_action.triggered.connect(self.keep_clicked.emit)
        self.trash_action.triggered.connect(self.trash_clicked.emit)
        self.unsure_action.triggered.connect(self.unsure_clicked.emit)
        # Predict & Sort, Export CSV, Reset Model
        self.predict_sort_action = QAction("Predict & Sort", self)
        self.export_csv_action = QAction("Export CSV", self)
        self.reset_model_action = QAction("Reset Personal Model", self)
        self.addAction(self.predict_sort_action)
        self.addAction(self.export_csv_action)
        self.addAction(self.reset_model_action)
        self.predict_sort_action.triggered.connect(self.predict_sort_clicked.emit)
        self.export_csv_action.triggered.connect(self.export_csv_clicked.emit)
        self.reset_model_action.triggered.connect(self.reset_model_clicked.emit)
        # Zoom selector
        self.zoom_selector = QComboBox(self)
        self.zoom_selector.addItems(["80", "120", "160", "200", "240"])
        self.zoom_selector.setCurrentText("160")
        self.zoom_selector.setToolTip("Zoom (cell size)")
        self.addWidget(self.zoom_selector)
        self.zoom_selector.currentTextChanged.connect(self._on_zoom_changed)
    def _on_zoom_changed(self, value):
        self.zoom_changed.emit(int(value))
