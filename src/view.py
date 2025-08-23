from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QProgressBar, QGridLayout, QScrollArea, QLabel, QComboBox, QHBoxLayout, QPushButton, QLineEdit
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import os

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

        # Rating UI
        self.rating_layout = QHBoxLayout()
        self.rating_stars = []
        for i in range(1, 6):
            btn = QPushButton("☆")
            btn.setFixedWidth(28)
            btn.setFlat(True)
            btn.clicked.connect(lambda _, n=i: self._on_rating_clicked(n))
            self.rating_layout.addWidget(btn)
            self.rating_stars.append(btn)
        self.layout.addLayout(self.rating_layout)
        # Tags UI
        self.tags_layout = QHBoxLayout()
        self.tags_label = QLabel("Tags:")
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("comma-separated tags")
        self.tags_edit.editingFinished.connect(self._on_tags_edited)
        self.tags_layout.addWidget(self.tags_label)
        self.tags_layout.addWidget(self.tags_edit)
        self.layout.addLayout(self.tags_layout)

        self.label_refs = {}
        self.selected_filename = None
        self._connect_signals()

    def _connect_signals(self):
        self.viewmodel.images_changed.connect(self._on_images_changed)
        self.viewmodel.image_added.connect(self._on_image_added)
        self.viewmodel.exif_changed.connect(self._on_exif_changed)
        self.viewmodel.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self.viewmodel.progress_changed.connect(self._on_progress_changed)
        self.viewmodel.selected_image_changed.connect(self._on_selected_image_changed)
        self.viewmodel.has_selected_image_changed.connect(self._on_has_selected_image_changed)
        self.viewmodel.rating_changed.connect(self._on_rating_changed)
        self.viewmodel.tags_changed.connect(self._on_tags_changed)

    def _on_images_changed(self, images):
        # Only clear grid if images list is empty (full reload)
        if not images:
            for label in self.label_refs.values():
                label.deleteLater()
            self.label_refs.clear()
        self.selected_filename = None
        self.open_btn.setEnabled(False)
        self._on_rating_changed(0)
        self._on_tags_changed([])

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
        # Selection highlight
        self._update_label_highlight(label, filename)

    def _on_label_clicked(self, filename):
        self.selected_filename = filename
        self._update_all_highlights()
        self.viewmodel.select_image(filename)

    def _update_label_highlight(self, label, filename):
        if hasattr(self, 'selected_filename') and filename == self.selected_filename:
            label.setStyleSheet("border: 3px solid #0078d7; background: #e6f0fa;")
        else:
            label.setStyleSheet("")

    def _update_all_highlights(self):
        for (row, col), label in self.label_refs.items():
            filename = label.toolTip()
            self._update_label_highlight(label, filename)

    def _on_selected_image_changed(self, path):
        # Update highlight when selection changes from viewmodel
        if path:
            filename = os.path.basename(path)
            self.selected_filename = filename
            self._update_all_highlights()

    def resizeEvent(self, event):
        # Responsive grid: recalculate columns
        width = self.scroll_area.viewport().width()
        new_cols = max(1, width // (self.thumb_size + 16))
        if new_cols != self.images_per_row:
            self.images_per_row = new_cols
            self._relayout_grid()
        super().resizeEvent(event)

    def _relayout_grid(self):
        # Remove all widgets from grid
        for label in self.label_refs.values():
            self.grid_layout.removeWidget(label)
        # Re-add in new layout
        images = list(self.label_refs.values())
        for idx, label in enumerate(images):
            row = idx // self.images_per_row
            col = idx % self.images_per_row
            self.grid_layout.addWidget(label, row, col)

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

    def _on_rating_clicked(self, n):
        self.viewmodel.set_rating(n)

    def _on_rating_changed(self, rating):
        for i, btn in enumerate(self.rating_stars, 1):
            btn.setText("★" if i <= rating else "☆")

    def _on_tags_edited(self):
        tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        self.viewmodel.set_tags(tags)

    def _on_tags_changed(self, tags):
        self.tags_edit.setText(", ".join(tags))
