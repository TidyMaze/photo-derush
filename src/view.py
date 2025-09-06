from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTextEdit, QProgressBar, QGridLayout, QScrollArea, QLabel, QHBoxLayout, QPushButton, QDialog, QLineEdit
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from PySide6.QtCore import Qt
import os

class FullscreenDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        layout = QHBoxLayout(self)
        for path in image_paths:
            # Ensure path is a string and log the type for debugging
            import logging
            if isinstance(path, list):
                logging.error(f"Path is a list instead of string: {path}, taking first element")
                path = path[0] if path else ""
            elif not isinstance(path, (str, os.PathLike)):
                logging.error(f"Path is not a string or PathLike: {type(path)} = {path}")
                path = str(path)

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
                except Exception as e:
                    logging.error(f"Failed to open image in viewer: {e}")
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

        # Left: grid (filters removed)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.main_layout.addWidget(self.left_widget, stretch=3)

        # Grid area
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

        # Keep/Trash buttons
        self.label_layout = QHBoxLayout()
        self.keep_btn = QPushButton("Keep")
        self.trash_btn = QPushButton("Trash")
        self.label_layout.addWidget(self.keep_btn)
        self.label_layout.addWidget(self.trash_btn)
        self.side_layout.addLayout(self.label_layout)

        # Fullscreen/Compare button
        self.fullscreen_btn = QPushButton("Fullscreen/Compare")
        self.fullscreen_btn.setEnabled(True)
        self.fullscreen_btn.clicked.connect(self._on_fullscreen_clicked)
        self.side_layout.addWidget(self.fullscreen_btn)

        self.label_refs = {}
        self.selected_filename = None  # Kept for backward compatibility
        self._connect_signals()

        self.keep_btn.clicked.connect(lambda: self.viewmodel.set_label("keep"))
        self.trash_btn.clicked.connect(lambda: self.viewmodel.set_label("trash"))

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
        self.viewmodel.label_changed.connect(self._on_label_changed)

    def _on_images_changed(self, images):
        import logging
        logging.info(f"PhotoView._on_images_changed: images={images[:5]}... total={len(images)}")
        # Only clear grid and reset state; do not repopulate here
        for label in self.label_refs.values():
            label.deleteLater()
        self.label_refs.clear()
        self.selected_filename = None
        self.selected_filenames = set()
        self.open_btn.setEnabled(False)
        self._on_rating_changed(0)
        self._on_tags_changed([])
        self._on_label_changed(None, None)
        # Do not repopulate grid here; images will be added incrementally via _on_image_added

    def _on_image_added(self, filename, idx):
        import logging
        logging.info(f"PhotoView._on_image_added: filename={filename}, idx={idx}")
        row = idx // self.images_per_row
        col = idx % self.images_per_row
        label = QLabel()
        label.setFixedSize(self.thumb_size, self.thumb_size)
        label.setScaledContents(True)
        full_path = self.viewmodel.model.get_image_path(filename)

        # Ensure full_path is always a string
        if isinstance(full_path, list):
            logging.error(f"get_image_path returned list instead of string: {full_path}, taking first element")
            full_path = full_path[0] if full_path else ""
        elif not isinstance(full_path, (str, os.PathLike)):
            logging.error(f"get_image_path returned unexpected type: {type(full_path)} = {full_path}")
            full_path = str(full_path)

        label.setToolTip(full_path)  # Use full path for tooltip
        label.mousePressEvent = lambda e, f=filename, l=label: self._on_label_clicked(e, f, l)
        self.grid_layout.addWidget(label, row, col)
        logging.info(f"Added QLabel for {filename} at row={row}, col={col}")
        self.label_refs[(row, col)] = label
        self.viewmodel.load_thumbnail(filename)
        self._update_label_highlight(label, full_path)
        self.grid_widget.update()
        self.grid_layout.update()
        logging.info(f"grid_widget size: {self.grid_widget.size()}, grid_layout count: {self.grid_layout.count()}")

    def _on_label_clicked(self, event, filename, label):
        full_path = self.viewmodel.model.get_image_path(filename)

        # Ensure full_path is always a string
        if isinstance(full_path, list):
            import logging
            logging.error(f"get_image_path returned list in click handler: {full_path}, taking first element")
            full_path = full_path[0] if full_path else ""
        elif not isinstance(full_path, (str, os.PathLike)):
            import logging
            logging.error(f"get_image_path returned unexpected type in click handler: {type(full_path)} = {full_path}")
            full_path = str(full_path)

        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier or modifiers & Qt.KeyboardModifier.MetaModifier:
            if full_path in self.selected_filenames:
                self.selected_filenames.remove(full_path)
            else:
                self.selected_filenames.add(full_path)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier and self.selected_filenames:
            all_files = [label.toolTip() for label in self.label_refs.values()]

            # Validate that all tooltips are strings
            import logging
            for i, tooltip in enumerate(all_files):
                if isinstance(tooltip, list):
                    logging.error(f"Tooltip at index {i} is a list: {tooltip}, converting to string")
                    all_files[i] = tooltip[0] if tooltip else ""
                elif not isinstance(tooltip, (str, os.PathLike)):
                    logging.error(f"Tooltip at index {i} is unexpected type: {type(tooltip)} = {tooltip}")
                    all_files[i] = str(tooltip)

            last = None
            if self.selected_filenames:
                last = all_files.index(sorted(self.selected_filenames, key=lambda x: all_files.index(x))[-1])
            curr = all_files.index(full_path)
            if last is not None:
                rng = range(min(last, curr), max(last, curr) + 1)
                for i in rng:
                    file_path = all_files[i]
                    # Double-check the file_path before adding
                    if isinstance(file_path, (str, os.PathLike)) and file_path:
                        self.selected_filenames.add(file_path)
        else:
            self.selected_filenames = {full_path}
        self.selected_filename = full_path
        self._update_all_highlights()
        self.viewmodel.select_image(filename)

    def _update_label_highlight(self, label, full_path):
        if full_path in getattr(self, 'selected_filenames', set()):
            label.setStyleSheet("border: 3px solid #0078d7; background: #e6f0fa;")
        else:
            label.setStyleSheet("")

    def _update_all_highlights(self):
        for (row, col), label in self.label_refs.items():
            full_path = label.toolTip()
            self._update_label_highlight(label, full_path)

    def _on_label_changed(self, filename, label):
        if not self.selected_filenames:
            return

        files_to_update = [os.path.basename(f) for f in self.selected_filenames]

        for (row, col), thumb_label in self.label_refs.items():
            f_path = thumb_label.toolTip()
            f_name = os.path.basename(f_path)
            if f_name in files_to_update:
                details = self.viewmodel.model.get_image_details(f_name)
                current_label = details.get('label') if details else None
                self._update_label_icon(thumb_label, current_label)

    def _update_label_icon(self, thumb_label, label):
        if not hasattr(thumb_label, 'original_pixmap') or thumb_label.original_pixmap.isNull():
            return

        pixmap = thumb_label.original_pixmap.copy()
        painter = QPainter(pixmap)

        icon_char = ""
        color = QColor()

        if label == "keep":
            icon_char = "✓"
            color = QColor("green")
        elif label == "trash":
            icon_char = "✗"
            color = QColor("red")

        if icon_char:
            painter.setPen(color)
            font = painter.font()
            font.setPointSize(32)
            font.setBold(True)
            painter.setFont(font)

            rect = pixmap.rect()
            flags = Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight

            # Add a small margin
            rect.adjust(-10, 10, -10, 10)

            painter.drawText(rect, flags, icon_char)

        painter.end()
        thumb_label.setPixmap(pixmap)

    def _on_selected_image_changed(self, path):
        if path:
            self.selected_filename = path
            if path not in self.selected_filenames:
                self.selected_filenames = {path}
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
                        data = thumb.tobytes()
                        w, h = thumb.size
                        img_format = getattr(QImage, 'Format_RGBA8888', QImage.Format.Format_ARGB32)
                        qimg = QImage(data, w, h, img_format)
                        pixmap = QPixmap.fromImage(qimg)
                        logging.info(f"PIL Image thumbnail for {path} is valid: {not pixmap.isNull()}")

                    scaled_pixmap = pixmap.scaled(self.thumb_size, self.thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    label.original_pixmap = scaled_pixmap
                    label.setPixmap(scaled_pixmap)

                    filename = os.path.basename(path)
                    details = self.viewmodel.model.get_image_details(filename)
                    if details and 'label' in details:
                        self._update_label_icon(label, details['label'])

                    logging.info(f"Set pixmap for label {path}")
                else:
                    logging.warning(f"No thumbnail for {path}")
                break

    def _on_has_selected_image_changed(self, has_selection: bool):
        self.open_btn.setEnabled(has_selection)

    def _on_fullscreen_clicked(self):
        # Show selected image(s) in fullscreen/compare dialog
        selected = [f for f in self.selected_filenames]
        if not selected:
            return
        dlg = FullscreenDialog(selected, self)
        dlg.exec()

    def closeEvent(self, event):
        """Handle the window close event to ensure threads are cleaned up."""
        import logging
        if self.viewmodel._loader_thread and self.viewmodel._loader_thread.isRunning():
            logging.info("Close event: Loader thread is running. Initiating graceful shutdown.")
            # Connect the thread's finished signal to the window's close method
            self.viewmodel._loader_thread.finished.connect(self.close)
            # Ask the thread to stop
            self.viewmodel.cleanup()
            # Ignore the current close event, the window will be closed by the finished signal
            event.ignore()
        else:
            logging.info("Close event: Loader thread not running. Accepting close event.")
            event.accept()
