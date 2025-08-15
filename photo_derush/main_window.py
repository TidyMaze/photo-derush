from PySide6.QtWidgets import QMainWindow, QStatusBar, QSplitter, QWidget, QApplication
from .toolbar import SettingsToolbar
from .info_panel import InfoPanel
from .image_grid import ImageGrid

class LightroomMainWindow(QMainWindow):
    def __init__(self, image_paths, directory, get_sorted_images, image_info=None):
        super().__init__()
        self.setWindowTitle("Photo Derush (Qt)")
        self.resize(1400, 800)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.toolbar = SettingsToolbar(self)
        self.addToolBar(self.toolbar)
        self.info_panel = InfoPanel()
        self.splitter = QSplitter()
        self.setCentralWidget(self.splitter)
        self.left_panel = QWidget()
        self.sort_by_group = False
        self.image_info = image_info or {}
        def get_sorted_images():
            if self.sort_by_group and self.image_info:
                # Sort by group, then by filename
                def group_key(img):
                    info = self.image_info.get(img, {})
                    group = info.get("group")
                    return (group if group is not None else 999999, img)
                return sorted(image_paths, key=group_key)
            return image_paths
        self.image_grid = ImageGrid(image_paths, directory, self.info_panel, self.status, get_sorted_images, image_info=image_info)
        self.splitter.addWidget(self.image_grid)
        self.splitter.addWidget(self.info_panel)
        self.splitter.setSizes([1000, 400])
        self.toolbar.zoom_changed.connect(self.image_grid.set_cell_size)
        self.toolbar.sort_by_group_action.toggled.connect(self.on_sort_by_group_toggled)
    def on_sort_by_group_toggled(self, checked):
        self.sort_by_group = checked
        self.image_grid.populate_grid()
    def closeEvent(self, event):
        QApplication.quit()
        super().closeEvent(event)
