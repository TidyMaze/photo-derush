from PySide6.QtWidgets import QToolBar
from PySide6.QtGui import QAction

class SettingsToolbar(QToolBar):
    def __init__(self, parent=None):
        super().__init__("Settings Toolbar", parent)
        self.sort_by_group_action = QAction("Sort by group", self)
        self.sort_by_group_action.setCheckable(True)
        self.addAction(self.sort_by_group_action)

