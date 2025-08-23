from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QProgressBar, QGridLayout, QScrollArea, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class PhotoView(QMainWindow):
    def __init__(self, viewmodel, thumb_size=128, images_per_row=10):
        super().__init__()
        self.viewmodel = viewmodel
        self.thumb_size = thumb_size
        self.images_per_row = images_per_row
        self.setWindowTitle("Photo App - Image Browser")
        self.resize(1000, 700)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.grid_widget)
        self.layout.addWidget(self.scroll_area)

        self.exif_view = QTextEdit()
        self.exif_view.setReadOnly(True)
        self.exif_view.setMinimumHeight(120)
        self.exif_view.setPlaceholderText("Select an image to view EXIF data.")
        self.layout.addWidget(self.exif_view)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        self.label_refs = {}
        self._connect_signals()

    def _connect_signals(self):
        self.viewmodel.images_changed.connect(self._on_images_changed)
        self.viewmodel.image_added.connect(self._on_image_added)
        self.viewmodel.exif_changed.connect(self._on_exif_changed)
        self.viewmodel.thumbnail_loaded.connect(self._on_thumbnail_loaded)

    def _on_images_changed(self, images):
        # Only clear grid if images list is empty (full reload)
        if not images:
            for label in self.label_refs.values():
                label.deleteLater()
            self.label_refs.clear()

    def _on_image_added(self, filename, idx):
        row = idx // self.images_per_row
        col = idx % self.images_per_row
        label = QLabel()
        label.setFixedSize(self.thumb_size, self.thumb_size)
        label.setScaledContents(True)
        label.setToolTip(filename)
        label.mousePressEvent = lambda e, f=filename: self.viewmodel.select_image(f)
        self.grid_layout.addWidget(label, row, col)
        self.label_refs[(row, col)] = label
        self.viewmodel.load_thumbnail(filename)

    def _on_thumbnail_loaded(self, path, thumb):
        # Find label by filename
        for (row, col), label in self.label_refs.items():
            if label.toolTip() == path or label.toolTip() in path:
                if thumb:
                    if isinstance(thumb, QImage):
                        pixmap = QPixmap.fromImage(thumb)
                    else:
                        # PIL Image
                        data = thumb.tobytes()
                        w, h = thumb.size
                        # Use Format_RGBA8888 if available, else fallback
                        img_format = getattr(QImage, 'Format_RGBA8888', QImage.Format_RGB32)
                        qimg = QImage(data, w, h, img_format)
                        pixmap = QPixmap.fromImage(qimg)
                    label.setPixmap(pixmap.scaled(self.thumb_size, self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                break

    def _on_exif_changed(self, exif):
        if not exif:
            self.exif_view.setText("No EXIF data found.")
        else:
            lines = [f"{k}: {v}" for k, v in sorted(exif.items())]
            self.exif_view.setText("\n".join(lines))
