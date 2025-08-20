from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt

class HoverEffectLabel(QLabel):
    clicked = Signal()
    doubleClicked = Signal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_selected = False
        self._base_style = "background: #444; border: 2px solid #444;"
        self._hover_style = "background: #666; border: 2px solid #888;"
        self.setStyleSheet(self._base_style)
    def set_selected(self, selected: bool):
        self._is_selected = selected
        if selected:
            self.setStyleSheet("background: red; border: 2px solid red;")
        else:
            self.setStyleSheet(self._base_style)
    def enterEvent(self, event):
        if not self._is_selected:
            self.setStyleSheet(self._hover_style)
        super().enterEvent(event)
    def leaveEvent(self, event):
        if not self._is_selected:
            self.setStyleSheet(self._base_style)
        super().leaveEvent(event)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)
