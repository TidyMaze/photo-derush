import sys
import logging
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QThread, Signal
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Photo App")
        self.resize(400, 300)
        self.list_widget = QListWidget()
        self.button = QPushButton("Select Directory")
        self.button.clicked.connect(self.select_directory)
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.list_widget)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.scanner = None
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.list_widget.clear()
            self.button.setEnabled(False)
            self.scanner = FileScanner(dir_path)
            self.scanner.files_found.connect(self.display_files)
            self.scanner.finished.connect(lambda: self.button.setEnabled(True))
            self.scanner.start()
    def display_files(self, files):
        self.list_widget.addItems(files)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

