from PySide6.QtWidgets import QToolBar, QComboBox
from PySide6.QtGui import QAction, QIcon, QColor, QImage, QPixmap
from PySide6.QtCore import Signal

class SettingsToolbar(QToolBar):
    zoom_changed = Signal(int)
    keep_clicked = Signal()
    trash_clicked = Signal()
    unsure_clicked = Signal()
    predict_sort_clicked = Signal()  # kept for backward compatibility (defaults to desc)
    predict_sort_desc_clicked = Signal()
    predict_sort_asc_clicked = Signal()
    export_csv_clicked = Signal()
    reset_model_clicked = Signal()
    def __init__(self, parent=None):
        super().__init__("Settings Toolbar", parent)
        self.sort_by_group_action = QAction(QIcon.fromTheme("view-sort-ascending"), "Sort by group", self)
        self.sort_by_group_action.setCheckable(True)
        self.sort_by_group_action.setToolTip("Sort images by group")
        self.addAction(self.sort_by_group_action)
        # Keep/Trash/Unsure actions with icons and tooltips
        self.keep_action = QAction(QIcon.fromTheme("emblem-favorite"), "Keep", self)
        self.keep_action.setToolTip("Mark as Keep")
        self.trash_action = QAction(QIcon.fromTheme("user-trash"), "Trash", self)
        self.trash_action.setToolTip("Mark as Trash")
        self.unsure_action = QAction(QIcon.fromTheme("dialog-question"), "Unsure", self)
        self.unsure_action.setToolTip("Mark as Unsure")
        self.addAction(self.keep_action)
        self.addAction(self.trash_action)
        self.addAction(self.unsure_action)
        self.keep_action.triggered.connect(self.keep_clicked.emit)
        self.trash_action.triggered.connect(self.trash_clicked.emit)
        self.unsure_action.triggered.connect(self.unsure_clicked.emit)
        # Predict & Sort actions
        self.predict_sort_action = QAction(QIcon.fromTheme("view-sort-descending"), "Predict & Sort (Desc)", self)
        self.predict_sort_action.setToolTip("Sort by predicted keep probability (descending)")
        self.predict_sort_asc_action = QAction(QIcon.fromTheme("view-sort-ascending"), "Predict & Sort (Asc)", self)
        self.predict_sort_asc_action.setToolTip("Sort by predicted keep probability (ascending)")
        self.addAction(self.predict_sort_action)
        self.addAction(self.predict_sort_asc_action)
        self.predict_sort_action.triggered.connect(self._emit_desc)
        self.predict_sort_asc_action.triggered.connect(self._emit_asc)
        # Export CSV, Reset Model
        self.export_csv_action = QAction(QIcon.fromTheme("document-save"), "Export CSV", self)
        self.export_csv_action.setToolTip("Export labels and features to CSV")
        self.reset_model_action = QAction(QIcon.fromTheme("edit-clear"), "Reset Personal Model", self)
        self.reset_model_action.setToolTip("Reset the personal model and event log")
        self.addAction(self.export_csv_action)
        self.addAction(self.reset_model_action)
        self.export_csv_action.triggered.connect(self.export_csv_clicked.emit)
        self.reset_model_action.triggered.connect(self.reset_model_clicked.emit)
        # Zoom selector
        self.zoom_selector = QComboBox(self)
        self.zoom_selector.addItems(["80", "120", "160", "200", "240"])
        self.zoom_selector.setCurrentText("160")
        self.zoom_selector.setToolTip("Zoom (cell size)")
        self.addWidget(self.zoom_selector)
        self.zoom_selector.currentTextChanged.connect(self._on_zoom_changed)
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'San Francisco', Arial, sans-serif;
                font-size: 13px;
                color: #f0f0f0;
            }
            QToolBar {
                background: #23272e;
                border-bottom: 1px solid #333;
                spacing: 6px;
                padding: 4px 8px;
                min-height: 36px;
            }
            QToolBar QToolButton {
                background: transparent;
                color: #fff;
                border: none;
                padding: 4px 10px;
                margin: 0 2px;
                border-radius: 6px;
            }
            QToolBar QToolButton:hover {
                background: #333;
                color: #fff;
            }
            QToolBar QComboBox {
                background: #23272e;
                color: #fff;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 2px 8px;
                min-width: 60px;
            }
            QToolBar QLabel, QToolBar QAbstractButton {
                color: #fff;
            }
        """)

        # Ensure all toolbar icons are white in dark mode
        def make_icon_white(action):
            icon = action.icon()
            if not icon.isNull():
                pixmap = icon.pixmap(32, 32)
                img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
                for y in range(img.height()):
                    for x in range(img.width()):
                        alpha = img.pixelColor(x, y).alpha()
                        img.setPixelColor(x, y, QColor(255, 255, 255, alpha))
                action.setIcon(QIcon(QPixmap.fromImage(img)))
        for action in [self.sort_by_group_action, self.keep_action, self.trash_action, self.unsure_action, self.predict_sort_action, self.predict_sort_asc_action, self.export_csv_action, self.reset_model_action]:
            make_icon_white(action)

    def _emit_desc(self):
        # backward compat emit old signal
        self.predict_sort_clicked.emit()
        self.predict_sort_desc_clicked.emit()

    def _emit_asc(self):
        self.predict_sort_asc_clicked.emit()

    def _on_zoom_changed(self, value):
        self.zoom_changed.emit(int(value))
