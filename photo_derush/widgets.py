from PySide6.QtWidgets import QLabel

class HoverEffectLabel(QLabel):
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

