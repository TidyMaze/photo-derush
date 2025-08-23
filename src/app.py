import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
import os

logging.basicConfig(level=logging.INFO)

class FileScanner(QThread):
    files_found = Signal(list)
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    def run(self):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        files = [f for f in os.listdir(self.directory)
                 if os.path.splitext(f)[1].lower() in image_exts]
        self.files_found.emit(files)

class ImageLoader(QThread):
    image_loaded = Signal(QPixmap)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        pixmap = QPixmap(self.path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_loaded.emit(pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Photo App")
        self.resize(400, 500)
        self.list_widget = QListWidget()
        self.button = QPushButton("Select Directory")
        self.button.clicked.connect(self.select_directory)
        self.preview_label = QLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(270)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.preview_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.scanner = None
        self.current_dir = None
        self.image_loader = None
        self.list_widget.currentItemChanged.connect(self.preview_selected_image)
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.list_widget.clear()
            self.button.setEnabled(False)
            self.current_dir = dir_path
            self.scanner = FileScanner(dir_path)
            self.scanner.files_found.connect(self.display_files)
            self.scanner.finished.connect(lambda: self.button.setEnabled(True))
            self.scanner.start()
    def display_files(self, files):
        self.list_widget.addItems(files)
    def preview_selected_image(self, current, previous):
        if not current or not self.current_dir:
            self.preview_label.setText("Select an image to preview")
            self.preview_label.setPixmap(QPixmap())
            return
        image_path = os.path.join(self.current_dir, current.text())
        self.preview_label.setText("Loading...")
        self.image_loader = ImageLoader(image_path)
        self.image_loader.image_loaded.connect(self.show_image)
        self.image_loader.start()
    def show_image(self, pixmap):
        if pixmap and not pixmap.isNull():
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setText("")
        else:
            self.preview_label.setText("Failed to load image")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
