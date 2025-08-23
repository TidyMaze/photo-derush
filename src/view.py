from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QProgressBar, QGridLayout, QScrollArea, QLabel, QComboBox, QHBoxLayout, QPushButton
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

        # File type filter UI
        self.filter_layout = QHBoxLayout()
        self.filetype_combo = QComboBox()
        self.filetype_combo.addItem("All", [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"])
        self.filetype_combo.addItem("JPG", [".jpg", ".jpeg"])
        self.filetype_combo.addItem("PNG", [".png"])
        self.filetype_combo.addItem("GIF", [".gif"])
        self.filetype_combo.addItem("BMP", [".bmp"])
        self.filetype_combo.addItem("TIFF", [".tiff"])
        self.filetype_combo.currentIndexChanged.connect(self._on_filetype_changed)
        self.filter_layout.addWidget(self.filetype_combo)
        self.layout.addLayout(self.filter_layout)

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

        self.open_btn = QPushButton("Open in Viewer")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._on_open_in_viewer)
        self.layout.addWidget(self.open_btn)

        self.label_refs = {}
        self._connect_signals()

    def _connect_signals(self):
        self.viewmodel.images_changed.connect(self._on_images_changed)
        self.viewmodel.image_added.connect(self._on_image_added)
        self.viewmodel.exif_changed.connect(self._on_exif_changed)
        self.viewmodel.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self.viewmodel.progress_changed.connect(self._on_progress_changed)
        self.viewmodel.selected_image_changed.connect(self._on_selected_image_changed)

    def _on_images_changed(self, images):
        # Only clear grid if images list is empty (full reload)
        if not images:
            for label in self.label_refs.values():
                label.deleteLater()
            self.label_refs.clear()
        self.open_btn.setEnabled(False)

    def _on_image_added(self, filename, idx):
        row = idx // self.images_per_row
        col = idx % self.images_per_row
        label = QLabel()
        label.setFixedSize(self.thumb_size, self.thumb_size)
        label.setScaledContents(True)
        label.setToolTip(filename)
        label.mousePressEvent = lambda e, f=filename: self._on_label_clicked(f)
        self.grid_layout.addWidget(label, row, col)
        self.label_refs[(row, col)] = label
        self.viewmodel.load_thumbnail(filename)

    def _on_label_clicked(self, filename):
        self.viewmodel.select_image(filename)
        # self.open_btn.setEnabled(True)  # Remove direct state management

    def _on_open_in_viewer(self):
        self.viewmodel.open_selected_in_viewer()

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

    def _on_progress_changed(self, current, total):
        if total > 0:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"Loading images: {current}/{total}")
            self.progress_bar.show()
        if current >= total:
            self.progress_bar.hide()

    def _on_filetype_changed(self, idx):
        exts = self.filetype_combo.currentData()
        self.viewmodel.set_file_types(exts)

    def _on_selected_image_changed(self, path):
        self.open_btn.setEnabled(bool(path))
