from PySide6.QtWidgets import QMainWindow, QStatusBar, QSplitter, QWidget, QApplication
from .toolbar import SettingsToolbar
from .info_panel import InfoPanel
from .image_grid import ImageGrid

class LightroomMainWindow(QMainWindow):
    def __init__(self, image_paths, directory, get_sorted_images):
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
        self.image_grid = ImageGrid(image_paths, directory, self.info_panel, self.status, get_sorted_images)
        self.splitter.addWidget(self.image_grid)
        self.splitter.addWidget(self.info_panel)
        self.splitter.setSizes([1000, 400])
    def closeEvent(self, event):
        QApplication.quit()
        super().closeEvent(event)

