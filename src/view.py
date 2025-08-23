from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QProgressBar, QGridLayout, QScrollArea, QLabel, QComboBox, QHBoxLayout, QPushButton, QLineEdit, QDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import os

class FullscreenDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        layout = QHBoxLayout(self)
        for path in image_paths:
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setScaledContents(True)
            if os.path.exists(path):
                from PIL import Image
                from PySide6.QtGui import QImage, QPixmap
                try:
                    img = Image.open(path)
                    img = img.convert("RGBA")
                    data = img.tobytes()
                    w, h = img.size
                    img_format = getattr(QImage, 'Format_RGBA8888', QImage.Format.Format_ARGB32)
                    qimg = QImage(data, w, h, img_format)
                    pixmap = QPixmap.fromImage(qimg)
                    label.setPixmap(pixmap.scaledToHeight(self.screen().size().height() - 100, Qt.TransformationMode.SmoothTransformation))
                except Exception:
                    label.setText("Failed to load image")
            else:
                label.setText("File not found")
            layout.addWidget(label)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_F, Qt.Key.Key_Space):
            self.close()
        else:
            super().keyPressEvent(event)

class PhotoView(QMainWindow):
    def __init__(self, viewmodel, thumb_size=128, images_per_row=10):
        super().__init__()
        self.viewmodel = viewmodel
        self.thumb_size = thumb_size
        self.images_per_row = images_per_row
        self.setWindowTitle("Photo App - Image Browser")
        self.resize(1000, 700)
        self.selected_filenames = set()  # Multi-selection support

        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # Left: grid and filter
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.main_layout.addWidget(self.left_widget, stretch=3)

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
        self.left_layout.addLayout(self.filter_layout)

        # Quick filter bar (by rating/tag/date)
        self.quick_filter_layout = QHBoxLayout()
        # Rating filter
        self.rating_filter_combo = QComboBox()
        self.rating_filter_combo.addItem("All", 0)
        for i in range(1, 6):
            self.rating_filter_combo.addItem(f"≥{i}★", i)
        self.rating_filter_combo.currentIndexChanged.connect(self._on_quick_filter_changed)
        self.quick_filter_layout.addWidget(self.rating_filter_combo)
        # Tag filter
        self.tag_filter_edit = QLineEdit()
        self.tag_filter_edit.setPlaceholderText("Filter by tag")
        self.tag_filter_edit.textChanged.connect(self._on_quick_filter_changed)
        self.quick_filter_layout.addWidget(self.tag_filter_edit)
        # Date filter
        self.date_filter_edit = QLineEdit()
        self.date_filter_edit.setPlaceholderText("YYYY-MM-DD")
        self.date_filter_edit.textChanged.connect(self._on_quick_filter_changed)
        self.quick_filter_layout.addWidget(self.date_filter_edit)
        self.left_layout.addLayout(self.quick_filter_layout)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.grid_widget)
        self.left_layout.addWidget(self.scroll_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.left_layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

        # Right: side panel
        self.side_panel = QWidget()
        self.side_layout = QVBoxLayout(self.side_panel)
        self.main_layout.addWidget(self.side_panel, stretch=2)

        self.exif_view = QTextEdit()
        self.exif_view.setReadOnly(True)
        self.exif_view.setMinimumHeight(120)
        self.exif_view.setPlaceholderText("Select an image to view EXIF data.")
        self.side_layout.addWidget(self.exif_view)

        self.open_btn = QPushButton("Open in Viewer")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._on_open_in_viewer)
        self.side_layout.addWidget(self.open_btn)

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
        self.side_layout.addLayout(self.rating_layout)
        # Tags UI
        self.tags_layout = QHBoxLayout()
        self.tags_label = QLabel("Tags:")
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("comma-separated tags")
        self.tags_edit.editingFinished.connect(self._on_tags_edited)
        self.tags_layout.addWidget(self.tags_label)
        self.tags_layout.addWidget(self.tags_edit)
        self.side_layout.addLayout(self.tags_layout)

        # Fullscreen/Compare button
        self.fullscreen_btn = QPushButton("Fullscreen/Compare")
        self.fullscreen_btn.setEnabled(True)
        self.fullscreen_btn.clicked.connect(self._on_fullscreen_clicked)
        self.side_layout.addWidget(self.fullscreen_btn)

        self.label_refs = {}
        self.selected_filename = None  # Kept for backward compatibility
        self._connect_signals()

    def _connect_signals(self):
        self.viewmodel.images_changed.connect(self._on_images_changed)
        self.viewmodel.image_added.connect(self._on_image_added)  # Re-enable this for incremental grid population
        self.viewmodel.exif_changed.connect(self._on_exif_changed)
        self.viewmodel.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self.viewmodel.progress_changed.connect(self._on_progress_changed)
        self.viewmodel.selected_image_changed.connect(self._on_selected_image_changed)
        self.viewmodel.has_selected_image_changed.connect(self._on_has_selected_image_changed)
        self.viewmodel.rating_changed.connect(self._on_rating_changed)
        self.viewmodel.tags_changed.connect(self._on_tags_changed)

    def _on_images_changed(self, images):
        import logging
        logging.info(f"PhotoView._on_images_changed: images={images}")
        # Only clear grid and reset state; do not repopulate here
        for label in self.label_refs.values():
            label.deleteLater()
        self.label_refs.clear()
        self.selected_filename = None
        self.selected_filenames = set()
        self.open_btn.setEnabled(False)
        self._on_rating_changed(0)
        self._on_tags_changed([])
        # Do not repopulate grid here; images will be added incrementally via _on_image_added

    def _on_image_added(self, filename, idx):
        import logging
        logging.info(f"PhotoView._on_image_added: filename={filename}, idx={idx}")
        row = idx // self.images_per_row
        col = idx % self.images_per_row
        label = QLabel()
        label.setFixedSize(self.thumb_size, self.thumb_size)
        label.setScaledContents(True)
        label.setToolTip(filename)
        label.mousePressEvent = lambda e, f=filename, l=label: self._on_label_clicked(e, f, l)
        self.grid_layout.addWidget(label, row, col)
        logging.info(f"Added QLabel for {filename} at row={row}, col={col}")
        self.label_refs[(row, col)] = label
        self.viewmodel.load_thumbnail(filename)
        self._update_label_highlight(label, filename)
        self.grid_widget.update()
        self.grid_layout.update()
        logging.info(f"grid_widget size: {self.grid_widget.size()}, grid_layout count: {self.grid_layout.count()}")

    def _on_label_clicked(self, event, filename, label):
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier:
            # Ctrl/Cmd+Click: toggle selection
            if filename in self.selected_filenames:
                self.selected_filenames.remove(filename)
            else:
                self.selected_filenames.add(filename)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier and self.selected_filenames:
            # Shift+Click: select range
            all_files = [label.toolTip() for label in self.label_refs.values()]
            last = None
            if self.selected_filenames:
                last = all_files.index(sorted(self.selected_filenames, key=lambda x: all_files.index(x))[-1])
            curr = all_files.index(filename)
            if last is not None:
                rng = range(min(last, curr), max(last, curr) + 1)
                for i in rng:
                    self.selected_filenames.add(all_files[i])
        else:
            # Single click: select only this
            self.selected_filenames = {filename}
        self.selected_filename = filename  # For backward compatibility
        self._update_all_highlights()
        self.viewmodel.select_image(filename)

    def _update_label_highlight(self, label, filename):
        if filename in getattr(self, 'selected_filenames', set()):
            label.setStyleSheet("border: 3px solid #0078d7; background: #e6f0fa;")
        else:
            label.setStyleSheet("")

    def _update_all_highlights(self):
        for (row, col), label in self.label_refs.items():
            filename = label.toolTip()
            self._update_label_highlight(label, filename)

    def _on_selected_image_changed(self, path):
        if path:
            filename = os.path.basename(path)
            self.selected_filename = filename
            if filename not in self.selected_filenames:
                self.selected_filenames = {filename}
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
        import logging
        exts = self.filetype_combo.currentData()
        logging.info(f"Filetype filter changed: {exts}")
        self.viewmodel.set_file_types(exts)

    def _on_quick_filter_changed(self):
        import logging
        rating_filter = self.rating_filter_combo.currentData()
        tag_filter = self.tag_filter_edit.text().strip()
        date_filter = self.date_filter_edit.text().strip()
        logging.info(f"Quick filter changed: rating={rating_filter}, tag='{tag_filter}', date='{date_filter}'")
        self.viewmodel.set_quick_filter(rating_filter, tag_filter, date_filter)

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

    def _on_open_in_viewer(self):
        self.viewmodel.open_selected_in_viewer()

    def _on_thumbnail_loaded(self, path, thumb):
        import logging
        logging.info(f"PhotoView._on_thumbnail_loaded: path={path}, thumb={'yes' if thumb else 'no'} type={type(thumb)}")
        for (row, col), label in self.label_refs.items():
            if label.toolTip() == path:
                if thumb:
                    if isinstance(thumb, QImage):
                        pixmap = QPixmap.fromImage(thumb)
                        logging.info(f"QImage thumbnail for {path} is valid: {not pixmap.isNull()}")
                    else:
                        # PIL Image
                        data = thumb.tobytes()
                        w, h = thumb.size
                        img_format = getattr(QImage, 'Format_RGBA8888', QImage.Format.Format_ARGB32)
                        qimg = QImage(data, w, h, img_format)
                        pixmap = QPixmap.fromImage(qimg)
                        logging.info(f"PIL Image thumbnail for {path} is valid: {not pixmap.isNull()}")
                    label.setPixmap(pixmap.scaled(self.thumb_size, self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    logging.info(f"Set pixmap for label {path}")
                else:
                    logging.warning(f"No thumbnail for {path}")
                break

    def _on_has_selected_image_changed(self, has_selection: bool):
        self.open_btn.setEnabled(has_selection)

    def _on_fullscreen_clicked(self):
        # Show selected image(s) in fullscreen/compare dialog
        selected = [self.viewmodel.model.get_image_path(f) for f in self.selected_filenames]
        if not selected:
            return
        dlg = FullscreenDialog(selected, self)
        dlg.exec()
