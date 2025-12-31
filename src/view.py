import logging
import os
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRunnable, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .flow_layout import FlowLayout
from .lazy_loader import LazyImageLoader
from .model_stats import _get_feature_names
from .vm_directory import change_directory


class ThumbRunnable(QObject, QRunnable):
    done = Signal(object, object, str)

    def __init__(self, img, size, path):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.img = img
        self.size = size
        self.path = path

    def run(self):
        try:
            logging.info(f"[ThumbRunnable] Starting for {self.path}")
            from PIL import Image

            pil = None
            try:
                if hasattr(self.img, "convert"):
                    pil = self.img.convert("RGBA")
                else:
                    pil = Image.fromarray(self.img).convert("RGBA")
            except Exception:
                try:
                    # attempt QPixmap -> QImage -> bytes -> PIL
                    from PySide6.QtGui import QPixmap

                    qpm = QPixmap.fromImage(self.img) if not isinstance(self.img, QPixmap) else self.img
                    qimg = qpm.toImage()
                    fmt = getattr(qimg, "Format_RGBA8888", getattr(qimg, "Format_ARGB32", None))
                    if fmt is not None:
                        qimg = qimg.convertToFormat(fmt)
                    w = qimg.width()
                    h = qimg.height()
                    ptr = qimg.bits()
                    pil = Image.frombytes("RGBA", (w, h), bytes(ptr))
                except Exception:
                    pil = None

            if pil is None:
                logging.warning(f"[ThumbRunnable] No PIL data for {self.path}, emitting done with None")
                self.done.emit(None, None, self.path)
                return

            # Preserve aspect ratio: use thumbnail() instead of resize()
            # This keeps original proportions, fitting within square bounds
            pil_thumb = pil.copy()
            pil_thumb.thumbnail((int(self.size), int(self.size)), resample=Image.LANCZOS)

            # Center on square canvas to maintain consistent label size
            canvas = Image.new("RGBA", (int(self.size), int(self.size)), (0, 0, 0, 0))
            offset_x = (int(self.size) - pil_thumb.width) // 2
            offset_y = (int(self.size) - pil_thumb.height) // 2
            canvas.paste(pil_thumb, (offset_x, offset_y))

            # Store metadata in PIL Image.info for later retrieval by _convert_to_pixmap
            canvas.info = {
                "thumb_offset_x": offset_x,
                "thumb_offset_y": offset_y,
                "thumb_width": pil_thumb.width,
                "thumb_height": pil_thumb.height,
            }

            logging.info(
                f"[ThumbRunnable] Thumbnail for {self.path}: actual {pil_thumb.width}x{pil_thumb.height}, offset ({offset_x},{offset_y})"
            )
            self.done.emit(canvas, None, self.path)
        except Exception:
            logging.exception("Thumbnail worker failed")
            self.done.emit(None, None, self.path)


# UI Constants (KISS: centralize magic values)
COLOR_KEEP = "#27ae60"
COLOR_TRASH = "#e74c3c"
COLOR_UNLABELED = "#95a5a6"
COLOR_SELECTED = "#0078d7"
COLOR_BG_SELECTED = "#e6f0fa"
BADGE_WIDTH = 24  # Smaller badge for thumbnails
BADGE_HEIGHT = 8   # Smaller badge for thumbnails
BADGE_MARGIN = 2


class PhotoView(QMainWindow):
    def __init__(self, viewmodel, thumb_size=200, images_per_row=8):
        super().__init__()
        self.viewmodel = viewmodel
        # persistable defaults
        self.default_thumb_size = thumb_size
        self.settings = QSettings("TidyMaze", "photo-derush")
        thumb_val = self.settings.value("thumb_size", thumb_size)
        try:
            # QSettings returns str or QVariant; always cast to str first
            self.thumb_size = int(str(thumb_val))
            # Migrate old default (128) to new default (200)
            if self.thumb_size == 128:
                self.thumb_size = 200
                self.settings.setValue("thumb_size", 200)
        except Exception:
            self.thumb_size = thumb_size
        self.images_per_row = images_per_row
        self.setWindowTitle("Photo App - Image Browser")
        self.resize(1000, 700)
        self.selected_filenames = set()
        self.label_refs = {}  # Maps (row, col) -> QLabel for widgets currently in grid
        self._label_to_grid_pos = {}  # Reverse mapping: QLabel -> (row, col) for O(1) lookup
        self._all_widgets = {}  # Maps filename -> QLabel for ALL widgets (including hidden)
        self._last_browser_state = None
        self._hover_tooltip = None  # Track current hover tooltip to prevent multiple windows
        self._active_toasts = []  # Track all active toast windows
        self._last_cols_per_row = None  # Cache column count to skip unnecessary relayouts
        self._last_filtered_images_order = None  # Track last filtered_images order to detect sort changes

        # Initialize lazy loader for non-blocking EXIF/thumbnail loading
        self.lazy_loader = LazyImageLoader(self.viewmodel.model, max_workers=4, cache_size=256)

        # Simple in-memory pixmap cache
        self._pixmap_cache = {}
        self._pixmap_cache_max = 256
        # Cache for converted QPixmaps (PIL->QPixmap conversions)
        self._converted_pixmap_cache = {}
        self._converted_pixmap_cache_max = 512

        self._build_ui()
        self._connect_signals()  # Connect signals AFTER UI is built

        # Purge stale caches on startup to prevent bad bounding boxes from previous runs
        # Don't clear detection cache on startup - preserve cache between runs
        self.purge_overlay_caches(clear_detection_cache=False)

    def purge_overlay_caches(self, clear_detection_cache=False):
        """Purge caches that might contain stale overlay data.
        
        Args:
            clear_detection_cache: If True, also remove detection cache file (default: False)
        """
        # Clear pixmap caches
        self._pixmap_cache.clear()
        self._converted_pixmap_cache.clear()

        # Clear base_pixmap from all labels (contains overlays)
        # Note: base_pixmap should NOT contain overlays, but clear it to force regeneration
        for label in self._all_widgets.values():
            if hasattr(label, "base_pixmap"):
                label.base_pixmap = None  # type: ignore[attr-defined]

        # Clear overlay cache from overlay_widget module if imported
        try:
            from src.overlay_widget import _overlay_cache
            _overlay_cache.clear()
        except ImportError:
            pass

        # Only clear detection cache file if explicitly requested (e.g., on startup)
        if clear_detection_cache:
            try:
                from src import object_detection
                cache_path = object_detection.get_cache_path()
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    logging.info(f"Removed detection cache file: {cache_path}")
                    # Clear in-memory cache in viewmodel
                    if hasattr(self.viewmodel, "_cached_detection_cache"):
                        delattr(self.viewmodel, "_cached_detection_cache")
                    if hasattr(self.viewmodel, "_cached_detection_cache_mtime"):
                        delattr(self.viewmodel, "_cached_detection_cache_mtime")
            except Exception as e:
                logging.warning(f"Failed to remove detection cache: {e}")

        logging.info("Purged overlay caches" + (" (including detection cache)" if clear_detection_cache else ""))
        self._setup_shortcuts()
        # Debounce timer for selection-driven heavy updates
        self._selection_debounce_timer = QTimer()
        self._selection_debounce_timer.setSingleShot(True)
        self._selection_debounce_timer.setInterval(60)  # ms
        self._selection_debounce_timer.timeout.connect(self._on_selection_debounced)

    @property
    def _dpr(self):
        """Device pixel ratio (cached per access)."""
        try:
            return float(QGuiApplication.primaryScreen().devicePixelRatio() or 1.0)
        except Exception:
            return 1.0

    @property
    def _state(self):
        """Current browser state (shorthand)."""
        return getattr(self, "_last_browser_state", None)

    def _pixmap_cache_get(self, key):
        from src.cache_config import is_cache_disabled
        if is_cache_disabled():
            return None
        return self._pixmap_cache.get(key)

    def _pixmap_cache_set(self, key, value):
        from src.cache_config import is_cache_disabled
        if is_cache_disabled():
            return
        self._pixmap_cache[key] = value
        if len(self._pixmap_cache) > self._pixmap_cache_max:
            self._pixmap_cache.pop(next(iter(self._pixmap_cache)))

    def _connect_signals(self):
        """Connect viewmodel signals to view slots."""
        # Connect viewmodel signals to view slots
        self.viewmodel.images_changed.connect(self._on_images_changed)
        self.viewmodel.image_added.connect(self._on_image_added)
        self.viewmodel.selection_changed.connect(self._on_selection_model_changed)
        self.viewmodel.exif_changed.connect(self._on_exif_changed)
        self.viewmodel.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self.viewmodel.progress_changed.connect(self._on_progress_changed)
        self.viewmodel.selected_image_changed.connect(self._on_selected_image_changed)
        self.viewmodel.has_selected_image_changed.connect(self._on_has_selected_image_changed)
        self.viewmodel.task_started.connect(self._on_task_started)
        self.viewmodel.task_progress.connect(self._on_task_progress)
        self.viewmodel.task_finished.connect(self._on_task_finished)
        self.viewmodel.model_stats_changed.connect(self._on_model_stats_changed)
        self.viewmodel.directory_changed.connect(self._on_directory_changed)
        self.viewmodel.browser_state_changed.connect(self._on_browser_state_changed)
        # Refresh badges when grouping completes to show group badges
        if hasattr(self.viewmodel, "grouping_completed"):
            self.viewmodel.grouping_completed.connect(self._refresh_thumbnail_badges)
        self.viewmodel.object_detection_ready.connect(self._on_detection_ready)
        # Connect lazy loader signal
        # Note: Avoid double-connecting; view model already connects to thumbnail_loaded
        # self.lazy_loader._signals.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self.lazy_loader._signals.detection_done.connect(self._on_detection_done)
        # Show a short toast when labels change (used for auto-label notifications)
        try:
            self.viewmodel.label_changed.connect(self._on_label_changed)
        except Exception:
            logging.exception("Failed to connect label_changed to _on_label_changed")

    def _build_ui(self):
        """Build the main UI layout."""
        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout(self.central_widget)
        # Minimize margins to reduce wasted space
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)
        self.setCentralWidget(self.central_widget)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        """Build left panel with image grid and progress bars."""
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
        # Connect resize event to recalculate columns
        self.scroll_area.viewport().installEventFilter(self)
        self.left_layout.addWidget(self.scroll_area)
        # allow ctrl+wheel on the grid area to control zoom
        try:
            self.scroll_area.viewport().installEventFilter(self)
        except Exception:
            logging.exception("Failed to install event filter on scroll area")

        # Unified progress bar for all operations (image loading, training, etc.)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.left_layout.addWidget(self.progress_bar)
        self.progress_bar.hide()

    def _build_right_panel(self):
        """Build right panel with controls and info displays."""
        # Make the entire right panel scrollable to avoid crushed widgets
        self.side_panel = QWidget()
        # Fix width to match scroll area and prevent horizontal scrolling
        self.side_panel.setFixedWidth(350)
        self.side_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.side_layout = QVBoxLayout(self.side_panel)
        # Minimize margins to reduce wasted space
        self.side_layout.setContentsMargins(3, 3, 3, 3)
        self.side_layout.setSpacing(12)  # Increased vertical spacing between all field sets
        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        # Remove margins from scroll area to eliminate gap
        self.side_scroll.setContentsMargins(0, 0, 0, 0)
        # Fix width to prevent horizontal scrolling
        self.side_scroll.setFixedWidth(350)
        self.side_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Prevent horizontal scrolling in the side panel; only allow vertical scrolling
        try:
            self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.side_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        except Exception:
            logging.exception("Failed to set scroll bar policy")
        self.side_scroll.setWidget(self.side_panel)
        # Use stretch=0 so side panel only takes the space it needs (up to max width), eliminating right gap
        self.main_layout.addWidget(self.side_scroll, stretch=0)

        self._build_action_buttons()  # Keep/Trash buttons at top
        self._build_directory_selector()
        self._build_search_bar()
        self._build_status_bar()
        self._build_batch_toolbar()
        self._build_exif_view()
        self._build_object_detection_view()
        self._build_model_stats()


    def _build_directory_selector(self):
        """Build directory selection controls."""
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(6)  # Added spacing between directory label and switch button
        dir_label = QLabel("Directory:")
        dir_label.setStyleSheet("font-size: 9px;")
        self.dir_display = QLabel()
        self.dir_display.setStyleSheet("color: #909090; font-size: 9px;")
        self.dir_display.setWordWrap(True)
        self.dir_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.switch_dir_btn = QPushButton("📁 Switch")
        self.switch_dir_btn.clicked.connect(self._on_switch_directory)
        self.switch_dir_btn.setStyleSheet("""
            QPushButton {
                padding: 3px 6px;
                margin: 2px;
                border: 1px solid #4a5a6a;
                border-radius: 3px;
                background: #3a4a5a;
                color: #a0b0c0;
                font-weight: 500;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #4a5a6a;
                border-color: #5a6a7a;
            }
        """)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_display, stretch=1)
        dir_layout.addWidget(self.switch_dir_btn)
        
        # Wrap directory layout in a widget to add top/bottom margins
        dir_widget = QWidget()
        dir_widget_layout = QVBoxLayout(dir_widget)
        dir_widget_layout.setContentsMargins(0, 10, 0, 10)  # Increased top and bottom margins
        dir_widget_layout.setSpacing(0)
        dir_widget_layout.addLayout(dir_layout)
        
        self.side_layout.addWidget(dir_widget)
        self._update_directory_display()

    def _build_search_bar(self):
        """Build search bar for filtering images."""
        search_group = QGroupBox("Search")
        search_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #d0d0d0;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                margin-top: 4px;
                padding-top: 4px;
                padding-left: 2px;
                padding-right: 2px;
                padding-bottom: 4px;
                font-size: 9px;
                background: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
            }
        """)
        search_layout = QVBoxLayout(search_group)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by filename...")
        # Prevent search input from forcing panel width (match side panel max width)
        # Search input should adapt to panel width (word wrap handled by QLineEdit)
        self.search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_input.setStyleSheet("""
            QLineEdit {
                padding: 4px 8px;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                background: #3a3a3a;
                color: #e0e0e0;
                font-size: 9px;
            }
            QLineEdit:focus {
                border-color: #5a6a7a;
                outline: none;
            }
        """)
        self.search_input.textChanged.connect(self._on_search_changed)

        # Quick filter buttons
        quick_filters = QHBoxLayout()
        quick_filters.setSpacing(8)  # Increased spacing between filter buttons
        self.btn_unlabeled = QPushButton("Unlabeled")
        self.btn_keep = QPushButton("Keep")
        self.btn_trash = QPushButton("Trash")
        self.btn_unlabeled.clicked.connect(lambda: self._apply_quick_filter("unlabeled"))
        self.btn_keep.clicked.connect(lambda: self._apply_quick_filter("keep"))
        self.btn_trash.clicked.connect(lambda: self._apply_quick_filter("trash"))

        # Style filter buttons with Lightroom palette
        self.btn_unlabeled.setStyleSheet("""
            QPushButton {
                padding: 3px 6px;
                margin: 2px;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                background: #3a3a3a;
                color: #c0c0c0;
                font-weight: 500;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #4a4a4a;
                border-color: #5a5a5a;
            }
            QPushButton:pressed {
                background: #2a2a2a;
            }
        """)
        self.btn_keep.setStyleSheet("""
            QPushButton {
                padding: 3px 6px;
                margin: 2px;
                border: 1px solid #4a5a4a;
                border-radius: 3px;
                background: #3a4a3a;
                color: #a0c0a0;
                font-weight: 500;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #4a5a4a;
                border-color: #5a6a5a;
            }
            QPushButton:pressed {
                background: #2a3a2a;
            }
        """)
        self.btn_trash.setStyleSheet("""
            QPushButton {
                padding: 3px 6px;
                margin: 2px;
                border: 1px solid #5a4a4a;
                border-radius: 3px;
                background: #4a3a3a;
                color: #c0a0a0;
                font-weight: 500;
                font-size: 9px;
            }
            QPushButton:hover {
                background: #5a4a4a;
                border-color: #6a5a5a;
            }
            QPushButton:pressed {
                background: #3a2a2a;
            }
        """)

        quick_filters.addWidget(self.btn_unlabeled)
        quick_filters.addWidget(self.btn_keep)
        quick_filters.addWidget(self.btn_trash)

        # Hide manual labeled checkbox
        self.hide_manual_checkbox = QCheckBox("Hide manually labeled")
        self.hide_manual_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 9px;
                color: #c0c0c0;
                padding: 6px 4px;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                background: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background: #3a3a3a;
            }
            QCheckBox:hover {
                background: #2a2a2a;
                border-radius: 3px;
            }
        """)
        self.hide_manual_checkbox.stateChanged.connect(self._on_hide_manual_changed)

        search_layout.addWidget(self.search_input)
        search_layout.addLayout(quick_filters)
        search_layout.addWidget(self.hide_manual_checkbox)
        self.side_layout.addWidget(search_group)

        # Connect Ctrl+F to focus search
        QShortcut(QKeySequence.StandardKey.Find, self, self.search_input.setFocus)


    def _build_batch_toolbar(self):
        """Build batch operations toolbar (shown when multiple images selected)."""
        self.batch_toolbar = QWidget()
        batch_layout = QHBoxLayout(self.batch_toolbar)
        batch_layout.setContentsMargins(5, 5, 5, 5)

        self.batch_label = QLabel("Batch Operations:")
        self.batch_keep_btn = QPushButton("Label Keep")
        self.batch_trash_btn = QPushButton("Label Trash")
        self.batch_clear_btn = QPushButton("Clear Selection")

        self.batch_keep_btn.clicked.connect(self._on_batch_keep)
        self.batch_trash_btn.clicked.connect(self._on_batch_trash)
        self.batch_clear_btn.clicked.connect(self._on_batch_clear)

        batch_layout.addWidget(self.batch_label)
        batch_layout.addWidget(self.batch_keep_btn)
        batch_layout.addWidget(self.batch_trash_btn)
        batch_layout.addStretch()
        batch_layout.addWidget(self.batch_clear_btn)

        self.batch_toolbar.setStyleSheet("""
            QWidget {
                background: #3a4a5a;
                border: 1px solid #4a5a6a;
                border-radius: 4px;
            }
            QPushButton {
                padding: 6px 12px;
                margin: 3px;
                border-radius: 4px;
                font-weight: 500;
                background: #3a3a3a;
                color: #c0c0c0;
                border: 1px solid #4a4a4a;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        self.batch_toolbar.hide()
        self.side_layout.addWidget(self.batch_toolbar)

    def _on_batch_keep(self):
        """Apply 'keep' label to all selected images."""
        selected = list(self.selected_filenames)
        if not selected:
            return
        for path in selected:
            filename = os.path.basename(path)
            self.viewmodel.set_label("keep")
        self._show_toast(f"Labeled {len(selected)} images as Keep")

    def _on_batch_trash(self):
        """Apply 'trash' label to all selected images."""
        selected = list(self.selected_filenames)
        if not selected:
            return
        for path in selected:
            filename = os.path.basename(path)
            self.viewmodel.set_label("trash")
        self._show_toast(f"Labeled {len(selected)} images as Trash")

    def _on_batch_rating(self):
        """Open rating dialog for batch rating."""
        from PySide6.QtWidgets import QInputDialog
        rating, ok = QInputDialog.getInt(self, "Batch Rating", "Enter rating (0-5):", 0, 0, 5, 1)
        if ok:
            selected = list(self.selected_filenames)
            for path in selected:
                self.viewmodel.set_rating(rating)
            self._show_toast(f"Set rating {rating} for {len(selected)} images")

    def _on_batch_clear(self):
        """Clear selection."""
        self.viewmodel.selection_model.clear()
        self.selected_filenames = set()
        self._update_all_highlights()

    def _on_search_changed(self, text):
        """Handle search text change - filter images."""
        if not text.strip():
            # Clear filter if search is empty
            if self.viewmodel._filter_ctrl.active():
                self.viewmodel._filter_ctrl.clear()
                self.viewmodel._apply_filters()
            return

        # Simple search: check if text matches filename
        self.viewmodel._filter_ctrl.set_date("")  # Clear date filter

        # Apply filename filter through viewmodel
        filtered = []
        search_lower = text.lower()
        for fname in self.viewmodel.images:
            if search_lower in fname.lower():
                filtered.append(fname)

        # Update filtered images
        self.viewmodel._filtered_images = filtered
        self.viewmodel._apply_filters()

    def _apply_quick_filter(self, filter_type):
        """Apply quick filter preset."""
        self.search_input.clear()
        if filter_type == "unlabeled":
            self.viewmodel._filter_ctrl.set_batch(date="")
            # Filter to show only unlabeled
            filtered = []
            for fname in self.viewmodel.images:
                path = self.viewmodel.model.get_image_path(fname)
                if path:
                    label = self.viewmodel.model.get_state(path)
                    if not label:
                        filtered.append(fname)
            self.viewmodel._filtered_images = filtered
        elif filter_type == "keep":
            filtered = []
            for fname in self.viewmodel.images:
                path = self.viewmodel.model.get_image_path(fname)
                if path:
                    label = self.viewmodel.model.get_state(path)
                    if label == "keep":
                        filtered.append(fname)
            self.viewmodel._filtered_images = filtered
        elif filter_type == "trash":
            filtered = []
            for fname in self.viewmodel.images:
                path = self.viewmodel.model.get_image_path(fname)
                if path:
                    label = self.viewmodel.model.get_state(path)
                    if label == "trash":
                        filtered.append(fname)
            self.viewmodel._filtered_images = filtered

        self.viewmodel._apply_filters()

    def _on_hide_manual_changed(self, state):
        """Handle hide manual labeled checkbox state change."""
        hide = self.hide_manual_checkbox.isChecked()
        logging.info(f"[view] Hide manual checkbox changed: state={state}, isChecked={hide}")
        self.viewmodel.set_filter_hide_manual(hide)

    def _build_status_bar(self):
        """Build status bar with image statistics (similar to web app status cards)."""

        # Dataset status: two rows. First row: Total / Labeled / Unlabeled.
        # Second row: Detector / Device / Model info.
        self.status_group = QGroupBox("Dataset Status")
        self.status_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.status_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #d0d0d0;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                margin-top: 4px;
                padding-top: 4px;
                padding-left: 2px;
                padding-right: 2px;
                padding-bottom: 4px;
                font-size: 9px;
                background: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
            }
        """)

        # Status cards (no emojis for compact modern UI)
        self.total_images_label = QLabel("Total: 0")
        self.labeled_images_label = QLabel("Labeled: 0")
        self.unlabeled_images_label = QLabel("Unlabeled: 0")
        self.selected_count_label = QLabel("Selected: 0")
        # Detection model/device display (combined backend+model into single field)
        self.det_model_label = QLabel("Det: unknown")
        self.det_device_label = QLabel("Dev: unknown")
        # Classification model display
        self.class_model_label = QLabel("Class: unknown")
        # Embedding model display
        self.embed_model_label = QLabel("Embed: unknown")
        # Feature extraction display
        self.feature_backend_label = QLabel("Feat: unknown")

        # Style the main metric labels with Lightroom palette (low contrast, dark grays)
        self.total_images_label.setStyleSheet("""
            QLabel {
                background: #3a3a3a;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                padding: 2px 4px;
                font-weight: 600;
                font-size: 9px;
                color: #e0e0e0;
            }
        """)
        self.total_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.labeled_images_label.setStyleSheet("""
            QLabel {
                background: #3a4a3a;
                border: 1px solid #4a5a4a;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 600;
                font-size: 9px;
                color: #b0d0b0;
                min-width: 80px;
            }
        """)
        self.labeled_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labeled_images_label.setWordWrap(False)

        self.unlabeled_images_label.setStyleSheet("""
            QLabel {
                background: #4a4a3a;
                border: 1px solid #5a5a4a;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 600;
                font-size: 9px;
                color: #d0c0a0;
                min-width: 80px;
            }
        """)
        self.unlabeled_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unlabeled_images_label.setWordWrap(False)

        # Smaller, inline info labels with Lightroom palette
        info_style = """
            QLabel {
                color: #c0c0c0;
                padding: 3px 5px;
                font-size: 9px;
                background: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                font-weight: 500;
            }
        """
        for label in [self.det_model_label, self.det_device_label, self.class_model_label, self.embed_model_label, self.feature_backend_label]:
            label.setStyleSheet(info_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Enable word wrap for labels that might contain long text
            label.setWordWrap(True)

        # Arrange into multiple rows with spacing (compact)
        outer = QVBoxLayout()
        outer.setSpacing(6)  # Increased vertical spacing between rows
        outer.setContentsMargins(2, 4, 2, 4)  # Increased top/bottom margins
        
        # Row 1: Dataset stats
        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        top_row.addWidget(self.total_images_label)
        top_row.addWidget(self.labeled_images_label)
        top_row.addWidget(self.unlabeled_images_label)
        top_row.addWidget(self.selected_count_label)

        # Style selection counter with Lightroom palette
        self.selected_count_label.setStyleSheet("""
            QLabel {
                background: #3a4a5a;
                border: 1px solid #4a5a6a;
                border-radius: 2px;
                padding: 2px 4px;
                font-weight: 600;
                font-size: 9px;
                color: #a0b0c0;
            }
        """)
        self.selected_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Row 2: Object Detection info
        detection_row = QHBoxLayout()
        detection_row.setSpacing(4)
        detection_row.addWidget(self.det_model_label)
        detection_row.addWidget(self.det_device_label)
        
        # Row 3: Classification, Embeddings, Features
        ml_row = QHBoxLayout()
        ml_row.setSpacing(4)
        ml_row.addWidget(self.class_model_label)
        ml_row.addWidget(self.embed_model_label)
        ml_row.addWidget(self.feature_backend_label)
        
        outer.addLayout(top_row)
        outer.addLayout(detection_row)
        outer.addLayout(ml_row)
        self.status_group.setLayout(outer)

        self.side_layout.addWidget(self.status_group)
        self._update_status_bar()

    def _show_toast(self, message: str, timeout_ms: int = 1800):
        """Show a transient non-modal toast message - disabled to prevent window accumulation."""
        # Toast notifications disabled to prevent multiple window creation
        # Messages are logged instead
        logging.info(f"[Toast] {message}")
        pass

    def _close_all_toasts(self):
        """Close all active toast windows."""
        for toast in list(self._active_toasts):
            try:
                toast.hide()
                toast.deleteLater()
            except Exception:
                pass
        self._active_toasts.clear()

    def _update_status_bar(self):
        """Update the status bar with current image statistics."""
        try:
            total_images = len(self.viewmodel.images) if self.viewmodel else 0

            # Count labeled vs unlabeled images
            labeled_count = 0
            manual_count = 0
            auto_count = 0

            if self.viewmodel and hasattr(self.viewmodel, "model"):
                repo = getattr(self.viewmodel.model, "_repo", None)
                if repo:
                    for filename in self.viewmodel.images:
                        label = repo.get_state(filename)
                        if label:
                            labeled_count += 1
                            source = repo.get_label_source(filename)
                            if source == "manual":
                                manual_count += 1
                            elif source == "auto":
                                auto_count += 1

            unlabeled_count = total_images - labeled_count

            # Count selected images
            # OPTIMIZATION: selected_filenames is always set, use direct access
            selected_count = len(self.selected_filenames)

            # Update labels (no emojis for compact modern UI)
            self.total_images_label.setText(f"Total: {total_images}")
            self.labeled_images_label.setText(f"Labeled: {labeled_count}")
            self.unlabeled_images_label.setText(f"Unlabeled: {unlabeled_count}")
            self.selected_count_label.setText(f"Selected: {selected_count}")

            # Show all computation models/devices from viewmodel snapshot if available
            try:
                # Detection info (combined backend+model into single field)
                device = getattr(self._state, "detection_device", None) if self._state else None
                model = getattr(self._state, "detection_model", None) if self._state else None
                if device:
                    self.det_device_label.setText(f"Dev: {device}")
                if model and model != "unknown":
                    self.det_model_label.setText(f"Det: {model}")

                # Classification info
                class_type = getattr(self._state, "classification_model_type", None) if self._state else None
                class_path = getattr(self._state, "classification_model_path", None) if self._state else None
                if class_type != "unknown":
                    self.class_model_label.setText(f"Class: {class_type}")
                elif class_path != "unknown":
                    self.class_model_label.setText(f"Class: {class_path}")

                # Embedding info
                embed_model = getattr(self._state, "embedding_model", None) if self._state else None
                embed_device = getattr(self._state, "embedding_device", None) if self._state else None
                if embed_model != "unknown":
                    self.embed_model_label.setText(f"Embed: {embed_model} ({embed_device})")

                # Feature extraction info
                feature_backend = getattr(self._state, "feature_extraction_backend", None) if self._state else None
                if feature_backend != "unknown":
                    self.feature_backend_label.setText(f"Feat: {feature_backend}")

                # If state didn't include device/model (model not loaded yet), try fallback to object_detection module
                if not device or not model or device == "unknown" or model == "unknown":
                    try:
                        from . import object_detection
                        from .constants import YOLO_MODEL_NAME
                        import platform

                        try:
                            ob_device = object_detection.get_loaded_device()
                            ob_model = object_detection.get_loaded_model_name()
                        except Exception:
                            ob_device = None
                            ob_model = None
                        
                        # Show device - use loaded if available, otherwise determine expected using same logic as _load_model
                        if ob_device is not None:
                            self.det_device_label.setText(f"Dev: {str(ob_device)}")
                        elif device == "unknown":
                            # Use same device detection logic as object_detection._load_model()
                            try:
                                import torch
                                backend = getattr(object_detection, "DETECTION_BACKEND", "yolov8")
                                if torch.cuda.is_available():
                                    self.det_device_label.setText("Dev: cuda")
                                elif torch.backends.mps.is_available():
                                    # YOLOv8 on MPS on macOS is unstable, so force CPU (same logic as _load_model)
                                    if platform.system() == "Darwin" and backend == "yolov8":
                                        self.det_device_label.setText("Dev: cpu")
                                    else:
                                        self.det_device_label.setText("Dev: mps")
                                else:
                                    self.det_device_label.setText("Dev: cpu")
                            except ImportError:
                                self.det_device_label.setText("Dev: cpu")
                        
                        # Show model - use loaded if available, otherwise show expected
                        if ob_model:
                            self.det_model_label.setText(f"Det: {ob_model}")
                        elif model == "unknown":
                            self.det_model_label.setText(f"Det: {YOLO_MODEL_NAME}")
                    except Exception:
                        logging.exception("Failed to import object_detection for fallback detector/device")
                        pass
            except Exception:
                logging.exception("Failed to resolve detector/device for status bar")

            # Add tooltips with breakdown
            self.labeled_images_label.setToolTip(f"Manual: {manual_count}, Auto: {auto_count}")

        except Exception as e:
            # Be defensive in case of errors
            logging.exception(f"Failed to update status bar: {e}")

    def _build_exif_view(self):
        """Build enhanced EXIF data display."""
        exif_group = QGroupBox("Image Details")
        exif_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #d0d0d0;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                margin-top: 4px;
                padding-top: 4px;
                padding-left: 2px;
                padding-right: 2px;
                padding-bottom: 4px;
                font-size: 9px;
                background: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
            }
        """)
        exif_layout = QVBoxLayout(exif_group)
        exif_layout.setContentsMargins(2, 4, 2, 4)  # Increased top/bottom margins
        exif_layout.setSpacing(4)  # Increased spacing

        # Enhanced EXIF display with better formatting
        self.exif_view = QTextEdit()
        self.exif_view.setReadOnly(True)
        self.exif_view.setMinimumHeight(150)
        self.exif_view.setMaximumHeight(300)
        self.exif_view.setPlaceholderText("Select an image to view details.")
        # Enable word wrap to prevent long lines from forcing panel width
        self.exif_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.exif_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.exif_view.setStyleSheet("""
            QTextEdit {
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 9px;
                background: #2a2a2a;
                color: #d0d0d0;
                border: 1px solid #3a3a3a;
                border-radius: 2px;
                padding: 3px;
            }
        """)

        exif_layout.addWidget(self.exif_view)
        self.side_layout.addWidget(exif_group)

    def _build_object_detection_view(self):
        """Build object detection display."""
        self.object_detection_group = QGroupBox("Detected Objects")
        self.object_detection_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #d0d0d0;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                margin-top: 4px;
                padding-top: 4px;
                padding-left: 2px;
                padding-right: 2px;
                padding-bottom: 4px;
                font-size: 9px;
                background: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
            }
        """)
        self.object_detection_layout = QVBoxLayout()
        self.object_detection_layout.setContentsMargins(2, 4, 2, 4)  # Increased top/bottom margins
        self.object_detection_layout.setSpacing(4)  # Increased spacing
        self.object_detection_group.setLayout(self.object_detection_layout)

        # Replace plain text with a chip-style list for better UX
        self.object_detection_placeholder = QLabel("Select an image to view detected objects.")
        self.object_detection_placeholder.setStyleSheet("color: #808080; font-style: italic;")
        self.object_detection_layout.addWidget(self.object_detection_placeholder)

        # Use a FlowLayout so chips wrap horizontally and look nicer
        self.object_detection_list = QWidget()
        self.object_detection_list_layout = FlowLayout(self.object_detection_list, spacing=6)
        self.object_detection_list.setLayout(self.object_detection_list_layout)
        # Prefer wrapping instead of expanding horizontally
        self.object_detection_list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.object_detection_layout.addWidget(self.object_detection_list)

        self.side_layout.addWidget(self.object_detection_group)

    def _build_action_buttons(self):
        """Build action buttons (Open, Keep, Trash, Fullscreen)."""
        self.open_btn = QPushButton("Open in Viewer")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._on_open_in_viewer)
        self.open_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                margin: 2px;
                border: 1px solid #ced4da;
                border-radius: 3px;
                background: #ffffff;
                color: #495057;
                font-weight: 500;
                font-size: 9px;
            }
            QPushButton:hover:enabled {
                background: #f8f9fa;
                border-color: #adb5bd;
            }
            QPushButton:disabled {
                background: #e9ecef;
                color: #adb5bd;
            }
        """)
        self.side_layout.addWidget(self.open_btn)

        # Keep/Trash buttons - small and at top
        self.label_layout = QHBoxLayout()
        self.label_layout.setSpacing(8)  # Increased spacing between Keep/Trash buttons
        self.keep_btn = QPushButton("Keep")
        self.trash_btn = QPushButton("Trash")
        self.keep_btn.clicked.connect(lambda: self.viewmodel.set_label("keep"))
        self.trash_btn.clicked.connect(lambda: self.viewmodel.set_label("trash"))
        # Style buttons with Lightroom palette - less saturated colors
        self.keep_btn.setStyleSheet("""
            QPushButton {
                background: #4a6a4a;
                color: #d0e0d0;
                border: 1px solid #5a7a5a;
                border-radius: 2px;
                padding: 2px 5px;
                margin: 2px;
                font-weight: 500;
                font-size: 8px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #5a7a5a;
            }
            QPushButton:pressed {
                background: #3a5a3a;
            }
        """)
        self.trash_btn.setStyleSheet("""
            QPushButton {
                background: #6a4a4a;
                color: #e0d0d0;
                border: 1px solid #7a5a5a;
                border-radius: 2px;
                padding: 2px 5px;
                margin: 2px;
                font-weight: 500;
                font-size: 8px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #7a5a5a;
            }
            QPushButton:pressed {
                background: #5a3a3a;
            }
        """)
        self.label_layout.addWidget(self.keep_btn)
        self.label_layout.addWidget(self.trash_btn)
        self.side_layout.insertLayout(0, self.label_layout)  # Insert at top

        # Training controls
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setStyleSheet("""
            QPushButton {
                padding: 2px 6px;
                margin: 2px;
                border: 1px solid #5a6a7a;
                border-radius: 2px;
                background: #4a5a6a;
                color: #d0d0e0;
                font-weight: 500;
                font-size: 8px;
                min-height: 20px;
            }
            QPushButton:hover:enabled {
                background: #5a6a7a;
                border-color: #6a7a8a;
            }
            QPushButton:disabled {
                background: #3a3a3a;
                color: #707070;
                border-color: #4a4a4a;
            }
        """)
        self.cancel_train_btn = QPushButton("Cancel")
        self.cancel_train_btn.setStyleSheet("""
            QPushButton {
                padding: 2px 6px;
                margin: 2px;
                border: 1px solid #5a5a5a;
                border-radius: 2px;
                background: #4a4a4a;
                color: #c0c0c0;
                font-weight: 500;
                font-size: 8px;
                min-height: 20px;
            }
            QPushButton:hover:enabled {
                background: #5a5a5a;
                border-color: #6a6a6a;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #606060;
                border-color: #3a3a3a;
            }
        """)
        self.cancel_train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._on_train_clicked)
        self.cancel_train_btn.clicked.connect(self._on_cancel_train_clicked)
        train_layout = QHBoxLayout()
        train_layout.setSpacing(8)  # Increased spacing between Train/Cancel buttons
        train_layout.addWidget(self.train_btn)
        train_layout.addWidget(self.cancel_train_btn)
        self.side_layout.addLayout(train_layout)

    def _on_train_clicked(self):
        try:
            logging.info("[view] Train button clicked")
            if hasattr(self.viewmodel, "start_training"):
                logging.info("[view] Calling viewmodel.start_training()")
                self.viewmodel.start_training()
                self.train_btn.setEnabled(False)
                self.cancel_train_btn.setEnabled(True)
            else:
                logging.warning("[view] viewmodel has no start_training method")
        except Exception as e:
            logging.exception(f"[view] Error in _on_train_clicked: {e}")
            self._show_toast(f"Training failed: {e}", timeout_ms=3000)

    def _on_cancel_train_clicked(self):
        if hasattr(self.viewmodel, "cancel_training"):
            self.viewmodel.cancel_training()
            self.cancel_train_btn.setEnabled(False)
            self.train_btn.setEnabled(True)


    def _build_model_stats(self):
        """Build model performance statistics display."""
        self.model_stats_group = QGroupBox("Model Performance")
        self.model_stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #d0d0d0;
                border: 1px solid #4a4a4a;
                border-radius: 2px;
                margin-top: 4px;
                padding-top: 4px;
                padding-left: 2px;
                padding-right: 2px;
                padding-bottom: 4px;
                font-size: 9px;
                background: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 2px;
            }
        """)
        self.model_stats_layout = QVBoxLayout()
        self.model_stats_layout.setContentsMargins(2, 4, 2, 4)  # Increased top/bottom margins
        self.model_stats_layout.setSpacing(4)  # Increased spacing
        self.model_stats_group.setLayout(self.model_stats_layout)
        # Prevent the model stats group from forcing a wide minimum width
        self.model_stats_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        # Metrics in 2 rows (3 columns) to avoid truncation and use space better
        from PySide6.QtWidgets import QGridLayout
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(2)  # Reduced spacing
        metrics_grid.setContentsMargins(1, 1, 1, 1)  # Reduced margins
        
        self.metric_auc = QLabel("AUC: —")
        self.metric_precision = QLabel("Precision: —")
        self.metric_f1 = QLabel("F1: —")
        self.metric_loss = QLabel("Loss: —")
        self.metric_iterations = QLabel("Iter: —")
        self.metric_patience = QLabel("Pat: —")
        
        # Style all metrics with uniform font size and Lightroom palette
        for w in (self.metric_auc, self.metric_precision, self.metric_f1, self.metric_loss, self.metric_iterations, self.metric_patience):
            w.setStyleSheet("QLabel { font-weight: 600; padding: 2px 3px; font-size: 9px; color: #e0e0e0; background: transparent; }")
            w.setWordWrap(False)  # Prevent text wrapping
            w.setMinimumHeight(24)  # Ensure enough height for text
        
        # Arrange in 2 rows x 3 columns
        metrics_grid.addWidget(self.metric_auc, 0, 0)
        metrics_grid.addWidget(self.metric_precision, 0, 1)
        metrics_grid.addWidget(self.metric_f1, 0, 2)
        metrics_grid.addWidget(self.metric_loss, 1, 0)
        metrics_grid.addWidget(self.metric_iterations, 1, 1)
        metrics_grid.addWidget(self.metric_patience, 1, 2)
        
        # Set column stretch to use available space
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)
        metrics_grid.setColumnStretch(2, 1)

        self.model_stats_layout.addLayout(metrics_grid)

        # Top features list (compact)
        self.top_features_label = QLabel("Top features:")
        self.top_features_label.setStyleSheet("color: #c0c0c0; font-weight: 600; margin-top: 2px; font-size: 9px;")
        self.top_features_list = QWidget()
        # Use FlowLayout for compact, wrapped feature chips
        self.top_features_list_layout = FlowLayout(self.top_features_list, spacing=6)
        self.top_features_list.setLayout(self.top_features_list_layout)

        self.model_stats_layout.addWidget(self.top_features_label)
        self.model_stats_layout.addWidget(self.top_features_list)

        self.side_layout.addWidget(self.model_stats_group, stretch=1)

    def _clear_layout(self, layout):
        """Remove all widgets from a layout (helper)."""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        shortcuts = [
            (QKeySequence.StandardKey.Undo, self.viewmodel.undo),
            (QKeySequence.StandardKey.Redo, self.viewmodel.redo),
            (QKeySequence("K"), lambda: self.viewmodel.set_label("keep")),
            (QKeySequence("T"), lambda: self.viewmodel.set_label("trash")),
            (QKeySequence(Qt.Key.Key_Left), lambda: self._navigate_selection(-1)),
            (QKeySequence(Qt.Key.Key_Right), lambda: self._navigate_selection(1)),
            (QKeySequence(Qt.Key.Key_Up), lambda: self._navigate_selection(-1)),
            (QKeySequence(Qt.Key.Key_Down), lambda: self._navigate_selection(1)),
            (QKeySequence("?"), self._show_shortcuts_help),
        ]
        for seq, handler in shortcuts:
            QShortcut(seq, self, handler)

    def closeEvent(self, event):
        """Clean up resources when closing the window."""
        logging.info("[view] CloseEvent triggered, cleaning up all processes...")
        self._close_hover_tooltip()
        self._close_all_toasts()

        # Shutdown lazy loader
        try:
            if hasattr(self, 'lazy_loader'):
                self.lazy_loader.shutdown(wait=False)
        except Exception as e:
            logging.warning(f"[view] Error shutting down lazy loader: {e}")

        # Cleanup viewmodel (will handle TaskRunner, detection worker, multiprocessing pools, etc.)
        try:
            if hasattr(self, 'viewmodel'):
                self.viewmodel.cleanup()
        except Exception as e:
            logging.warning(f"[view] Error cleaning up viewmodel: {e}")


        super().closeEvent(event)
        logging.info("[view] CloseEvent complete")

    def _is_auto_labeled(self, filename: str) -> bool:
        """Check if a file is auto-labeled (vs manually labeled)."""
        repo = getattr(self.viewmodel.model, "_repo", None)
        if not repo:
            return False
        try:
            source = repo.get_label_source(filename)
            return str(source) == "auto"
        except Exception:
            return False

    def _on_images_changed(self, images):
        self._close_hover_tooltip()
        for label in self._all_widgets.values():
            label.deleteLater()
        self.label_refs.clear()
        self._all_widgets.clear()
        self.selected_filenames = set()
        self.open_btn.setEnabled(False)
        self._update_status_bar()

    def _calculate_images_per_row(self) -> int:
        """Calculate number of images per row based on available width."""
        try:
            # Get available width from scroll area viewport
            viewport_width = self.scroll_area.viewport().width()
            if viewport_width <= 0:
                # Fallback to default if viewport not ready
                return self.images_per_row

            # Account for spacing: (n-1) gaps between n items, plus some margin
            spacing = self.grid_layout.spacing()
            margin = 20  # Extra margin to prevent overflow
            available_width = viewport_width - margin

            # Calculate how many thumbnails fit
            thumb_with_spacing = self.thumb_size + spacing
            if thumb_with_spacing <= 0:
                return self.images_per_row

            cols = max(1, int(available_width / thumb_with_spacing))

            return cols
        except Exception:
            logging.exception("Failed to calculate images_per_row")
            return self.images_per_row

    def _relayout_grid(self):
        """Relayout all images in the grid based on current width."""
        if not self._all_widgets:
            return

        # Get current column count
        cols_per_row = self._calculate_images_per_row()

        # Get filtered images from viewmodel BEFORE skip check to detect sort order changes
        filtered_images = []
        if hasattr(self.viewmodel, 'current_filtered_images'):
            filtered_images = list(self.viewmodel.current_filtered_images())

        # Optimization: Skip relayout if column count hasn't changed AND widgets are already in layout
        # AND filtered_images order hasn't changed (sort order is the same)
        # BUT: Always run if widgets exist but aren't in layout (label_refs empty but _all_widgets not empty)
        widgets_in_layout = len(self.label_refs) > 0
        # Compare as tuples to handle list vs tuple comparison
        filtered_images_tuple = tuple(filtered_images) if filtered_images else None
        # Order changed if: (1) last order is None (first time), OR (2) last order exists and is different
        filtered_order_changed = (self._last_filtered_images_order is None or
                                   filtered_images_tuple != self._last_filtered_images_order)
        # Skip ONLY if: column count unchanged AND widgets in layout AND order hasn't changed
        if (self._last_cols_per_row == cols_per_row and
            self._last_cols_per_row is not None and
            widgets_in_layout and
            not filtered_order_changed):
            return
        self._last_cols_per_row = cols_per_row
        # Store filtered_images order to detect sort changes (already retrieved above)
        self._last_filtered_images_order = filtered_images_tuple

        # Collect widgets that should be visible based on filtered_images
        # CRITICAL: Iterate filtered_images in order to preserve uncertainty-based sorting
        items = []
        seen_labels = set()  # Track labels to prevent duplicates
        label_to_first_filename = {}  # Track which filename each label was first seen under
        if filtered_images:
            # Use filtered_images to determine visibility AND order (preserves uncertainty sort)
            for f_name in filtered_images:
                if f_name in self._all_widgets:
                    label = self._all_widgets[f_name]
                    if label in seen_labels:
                        first_filename = label_to_first_filename.get(label, "unknown")
                        logging.error(f"[DUPLICATE] Filename {f_name} maps to label {id(label)} already used for {first_filename}! Skipping duplicate.")
                        continue
                    seen_labels.add(label)
                    label_to_first_filename[label] = f_name
                    items.append(label)
        else:
            # Fallback: if no filtered_images, use ALL widgets (for initial display)
            # This ensures images are displayed even if filters haven't been applied yet
            # CRITICAL: Deduplicate - same label might be stored under multiple filenames (bug)
            label_to_first_filename = {}  # Track which filename each label was first seen under
            # Removed frequent logging.info - called on every relayout (performance optimization)
            # logging.info(f"[GRID] Collecting items from _all_widgets (has {len(self._all_widgets)} entries)")
            for filename, label in self._all_widgets.items():
                if label in seen_labels:
                    first_filename = label_to_first_filename.get(label, "unknown")
                    logging.error(f"[DUPLICATE] Label {id(label)} stored under both '{first_filename}' and '{filename}'! Using first occurrence only.")
                    continue
                seen_labels.add(label)
                label_to_first_filename[label] = filename
                items.append(label)
            # Removed frequent logging.info - called on every relayout (performance optimization)
            # logging.info(f"[GRID] Collected {len(items)} unique items (deduplicated from {len(self._all_widgets)} entries)")

        # Disable updates during relayout for better performance
        self.grid_widget.setUpdatesEnabled(False)
        try:
            # Rebuild grid layout
            # First, remove all widgets from layout (but don't delete them)
            # Keep parent to prevent widgets from becoming separate windows
            while self.grid_layout.count():
                item = self.grid_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.hide()  # Hide before removing to prevent window creation

            # Clear label_refs before re-adding (but widgets are preserved in _all_widgets)
            self.label_refs.clear()
            self._label_to_grid_pos.clear()  # Clear reverse mapping

            # Re-add widgets in new positions
            # Track which labels we've already added to prevent duplicates
            seen_labels = set()
            for idx, label in enumerate(items):
                # Check for duplicate labels (bug detection)
                if label in seen_labels:
                    label_filename = getattr(label, "_thumb_filename", "unknown")
                    logging.error(f"[DUPLICATE] Label {id(label)} with filename {label_filename} appears multiple times in grid! Skipping duplicate at position {idx}")
                    continue
                seen_labels.add(label)

                new_row = idx // cols_per_row
                new_col = idx % cols_per_row
                # Ensure widget has parent before adding to layout
                if label.parent() is None:
                    label.setParent(self.grid_widget)
                self.grid_layout.addWidget(label, new_row, new_col)
                # OPTIMIZATION: Only call show() if widget is not already visible
                if not label.isVisible():
                    label.show()
                self.label_refs[(new_row, new_col)] = label
                self._label_to_grid_pos[label] = (new_row, new_col)  # Update reverse mapping

                # Removed frequent logging.debug - called for every label positioning (performance optimization)
                # label_filename = getattr(label, "_thumb_filename", "unknown")
                # logging.debug(f"[GRID] Positioned label {label_filename} at ({new_row}, {new_col}), label_id={id(label)}")
                # If label doesn't have a pixmap, request thumbnail
                # Simple check: if label has no pixmap set via setPixmap, request thumbnail
                filename = getattr(label, "_thumb_filename", None)
                if filename:
                    # Check if label actually has a visible pixmap
                    try:
                        current_pixmap = label.pixmap()
                        if not current_pixmap or current_pixmap.isNull():
                            # No pixmap set - request thumbnail
                            logging.info(f"[THUMBNAIL] Re-requesting thumbnail for {filename} (label shown without pixmap)")
                            self.viewmodel.load_thumbnail(filename)
                        else:
                            logging.debug(f"[THUMBNAIL] Label {filename} already has pixmap, skipping re-request")
                    except Exception:
                        # If we can't check, request thumbnail anyway to be safe
                        logging.info(f"[THUMBNAIL] Re-requesting thumbnail for {filename} (could not check pixmap)")
                        self.viewmodel.load_thumbnail(filename)

            # Ensure hidden widgets are still tracked (but not in layout)
            # This allows them to reappear when filter changes
            for f_name, label in self._all_widgets.items():
                if label not in items:
                    # Widget should be hidden - ensure it's not in layout and is hidden
                    if label.parent() == self.grid_widget:
                        # Remove from layout if still there
                        self.grid_layout.removeWidget(label)
                    label.hide()
                    # Don't add to label_refs (it's not in the grid)
                    # But widget is preserved in memory via _all_widgets reference

            # Removed frequent logging.info - called on every relayout (performance optimization)
            # grid_summary = []
            # for (r, c), lbl in sorted(self.label_refs.items()):
            #     fn = getattr(lbl, "_thumb_filename", "unknown")
            #     grid_summary.append(f"({r},{c})={fn}")
            # logging.info(f"[GRID] Final grid state: {len(self.label_refs)} positions, {len(set(self.label_refs.values()))} unique labels. Positions: {', '.join(grid_summary[:9])}")
        finally:
            self.grid_widget.setUpdatesEnabled(True)

    def _on_image_added(self, filename, idx):
        # Calculate columns dynamically based on available width
        cols_per_row = self._calculate_images_per_row()
        # Update _last_cols_per_row to match current column count
        # This ensures relayout optimization works correctly
        if self._last_cols_per_row != cols_per_row:
            self._last_cols_per_row = cols_per_row

        row = idx // cols_per_row
        col = idx % cols_per_row
        label = QLabel(self.grid_widget)  # Ensure parent is set immediately
        # Set label size FIRST so overlay widgets can use correct geometry
        label.setFixedSize(self.thumb_size, self.thumb_size)

        # Add reusable bounding box overlay widget
        from src.bbox_overlay_widget import BoundingBoxOverlayWidget
        bbox_overlay = BoundingBoxOverlayWidget(label)
        bbox_overlay.setGeometry(0, 0, self.thumb_size, self.thumb_size)
        bbox_overlay.hide()  # Hide until detections are ready
        label._bbox_overlay = bbox_overlay  # type: ignore[attr-defined]

        # Add reusable badge overlay widget
        from src.badge_overlay_widget import BadgeOverlayWidget
        badge_overlay = BadgeOverlayWidget(label)
        badge_overlay.setGeometry(0, 0, self.thumb_size, self.thumb_size)
        badge_overlay.hide()  # Hide until badge data is ready
        label._badge_overlay = badge_overlay  # type: ignore[attr-defined]

        # Add group badge overlay widget
        from src.group_badge_widget import GroupBadgeWidget
        group_badge_overlay = GroupBadgeWidget(label)
        group_badge_overlay.setGeometry(0, 0, self.thumb_size, self.thumb_size)
        group_badge_overlay.hide()  # Hide until group data is ready
        label._group_badge_overlay = group_badge_overlay  # type: ignore[attr-defined]
        # Use scaled contents so Qt handles device pixel ratio automatically.
        # We provide square pixmaps exactly thumb_size x thumb_size.
        label.setScaledContents(True)
        full_path = self.viewmodel.model.get_image_path(filename)

        # Ensure full_path is always a string for tooltips
        if isinstance(full_path, list):
            logging.error(f"get_image_path returned list instead of string: {full_path}, taking first element")
            full_path = full_path[0] if full_path else ""
        elif not isinstance(full_path, (str, os.PathLike)):
            logging.error(f"get_image_path returned unexpected type: {type(full_path)} = {full_path}")
            full_path = str(full_path)

        label.setToolTip(str(full_path))

        # Show placeholder immediately so grid appears fast
        placeholder = QPixmap(self.thumb_size, self.thumb_size)
        placeholder.fill(QGuiApplication.palette().color(label.backgroundRole()).darker(120))
        label.setPixmap(placeholder)

        label.base_pixmap = None  # type: ignore[attr-defined]
        label._pil_image = None  # type: ignore[attr-defined]
        # Event handlers
        label._thumb_filename = filename  # type: ignore[attr-defined]
        # Use stored filename from label, not closure, to handle grid reordering
        label.mousePressEvent = lambda e, l=label: self._on_label_clicked(e, getattr(l, "_thumb_filename", ""), l)
        label.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        label.customContextMenuRequested.connect(
            lambda pos, l=label: self._show_thumb_context_menu(l, getattr(l, "_thumb_filename", ""), pos)
        )
        label.installEventFilter(self)

        def _dbl(e, f=filename):
            if e.type() == QEvent.Type.MouseButtonDblClick:  # type: ignore[attr-defined]
                # Double-click disabled - focus on labeling workflow
                pass

        label.mouseDoubleClickEvent = _dbl

        # Ensure widget is visible before adding to layout
        label.show()
        self.grid_layout.addWidget(label, row, col)
        self.label_refs[(row, col)] = label

        # Check if this filename already has a widget (shouldn't happen, but log if it does)
        existing_label = self._all_widgets.get(filename)
        if existing_label and existing_label != label:
            logging.warning(f"[THUMBNAIL] Filename {filename} already has a widget, overwriting. Old widget: {existing_label}, New widget: {label}")

        # Check for duplicate filename in grid (bug detection)
        if filename in self._all_widgets:
            existing_label = self._all_widgets[filename]
            if existing_label != label:
                logging.error(f"[DUPLICATE] Filename {filename} already in grid at different position! Existing label: {existing_label}, New label: {label}, row={row}, col={col}")

        # Check if this label is already used for a different filename (would cause duplicate images)
        for existing_filename, existing_label in self._all_widgets.items():
            if existing_label == label and existing_filename != filename:
                logging.error(f"[DUPLICATE] Label {id(label)} already used for {existing_filename}, now being reused for {filename} at ({row}, {col})! This will cause duplicate images. Removing from grid.")
                # Remove label from grid layout to prevent duplicate display
                self.grid_layout.removeWidget(label)
                if (row, col) in self.label_refs:
                    del self.label_refs[(row, col)]
                label.hide()
                return  # Don't add duplicate label

        self._all_widgets[filename] = label  # Track all widgets by filename
        # Removed frequent logging.info - called for every image (performance optimization)
        # logging.info(f"[GRID] Added image {filename} at position ({row}, {col}), label_id={id(label)}")

        if idx == 0:
            self.scroll_area.verticalScrollBar().setValue(0)

        self.viewmodel.load_thumbnail(filename)
        self._update_label_highlight(label, full_path)

    def _on_label_clicked(self, event, filename, label):
        """Handle thumbnail click with multi-select support."""
        modifiers = event.modifiers()
        path = self.viewmodel.model.get_image_path(filename)
        if not path:
            return

        # Multi-select: Ctrl+Click toggles, Shift+Click extends range
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            # Toggle selection
            self.viewmodel.selection_model.toggle(path)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Range selection
            ordered_paths = [self.viewmodel.model.get_image_path(f) for f in self.viewmodel.images]
            ordered_paths = [p for p in ordered_paths if p]  # Filter None
            self.viewmodel.selection_model.extend_range(path, ordered_paths)
        else:
            # Single selection (default)
            self.viewmodel.handle_selection_click(filename, modifiers)

        self.selected_filenames = set(self.viewmodel.selection_model.selected())
        self._update_all_highlights()

    def _paint_label_badge(self, pixmap, label_text, label_source, probability=None):
        """Paint a label badge on the top-left corner of the pixmap.

        Args:
            pixmap: QPixmap to paint on (should be at retina resolution)
            label_text: The label text ('keep' or 'trash')
            label_source: 'manual', 'auto', or 'predicted' to indicate source
            probability: Optional float (0.0-1.0) to show as percentage
        """
        if not pixmap or pixmap.isNull():
            return pixmap

        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QColor, QFont, QPainter, QPen

            # Get device pixel ratio to scale badge dimensions
            dpr = pixmap.devicePixelRatio() or 1.0

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Single-line format: "76.5% K" or "24.5% T" (1 decimal precision) if probability available
            # Otherwise just "✓ K" or "🤖 T" without percentage
            is_keep = label_text and label_text.lower() == "keep"
            letter = "K" if is_keep else "T"

            # Emoji indicator for source: ✓ manual, 🤖 auto, no emoji for predicted
            emoji = ""
            if label_source == "manual":
                emoji = "✓ "
            elif label_source == "auto":
                emoji = "🤖 "

            # Only show percentage if we have a valid prediction probability
            if probability is not None and probability == probability:  # Check for NaN
                pct = probability * 100 if is_keep else (1 - probability) * 100
                badge_text = f"{emoji}{pct:.1f}% {letter}"
            else:
                # No prediction available - show label only without percentage
                badge_text = f"{emoji}{letter}"

            # Background color: different for manual vs auto
            # Manual: darker/more saturated, Auto: lighter/less saturated
            if label_source == "manual":
                # Manual: deeper colors
                bg_color = QColor(27, 140, 77, 240) if is_keep else QColor(200, 50, 40, 240)
            elif label_source == "auto":
                # Auto: lighter colors
                bg_color = QColor(60, 200, 120, 200) if is_keep else QColor(250, 120, 100, 200)
            else:
                # Predicted: default colors
                bg_color = QColor(39, 174, 96, 220) if is_keep else QColor(231, 76, 60, 220)

            # Badge dimensions: scale from logical to physical pixels, but keep small relative to thumbnail
            # Pixmap is 128x128 logical = 256x256 physical
            # Use smaller badge size - about 12% of thumbnail width
            thumb_size_logical = pixmap.width()  # 128 logical pixels
            badge_w_logical = max(14, int(thumb_size_logical * 0.12))  # ~15 logical pixels
            badge_h_logical = max(5, int(badge_w_logical * 0.33))  # ~5 logical pixels, maintain aspect
            x = int(BADGE_MARGIN * dpr)
            y = int(BADGE_MARGIN * dpr)
            badge_w = int(badge_w_logical * dpr)
            badge_h = int(badge_h_logical * dpr)

            # Drop shadow for contrast
            shadow_offset = int(1 * dpr)
            painter.fillRect(x + shadow_offset, y + shadow_offset, badge_w, badge_h, QColor(0, 0, 0, 100))

            # Background with color already set based on source (manual/auto have different colors)
            painter.fillRect(x, y, badge_w, badge_h, bg_color)

            # No border needed - emoji and color distinguish manual/auto

            # White text with dark outline for readability
            # Font size: smaller for compact badge (scale with badge size)
            font = QFont()
            font.setPointSizeF(2.8)  # Smaller font size for compact badge
            font.setBold(True)
            painter.setFont(font)

            # Text outline (dark) then fill (white) - thinner outline
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
            painter.drawText(x, y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, badge_text)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x, y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, badge_text)

            painter.end()
            return pixmap
        except Exception:
            logging.exception("Failed to paint label badge")
            return pixmap


    def _extract_training_data(self, stats: dict) -> tuple[list[float] | None, list[float] | None, list[int] | None, list[int] | None]:
        """Extract training data from stats dict. Returns (train_loss, val_loss, epochs, counts)."""
        # Try nested history first
        hist = stats.get("training_history") or stats.get("history")
        if isinstance(hist, dict):
            train = hist.get("loss") or hist.get("train_loss") or hist.get("training_loss")
            val = hist.get("val_loss") or hist.get("validation_loss")
        else:
            train = stats.get("train_loss") or stats.get("loss") or stats.get("training_loss") or stats.get("train_losses")
            val = stats.get("val_loss") or stats.get("validation_loss") or stats.get("val_losses")

        # Normalize to lists
        def _to_float_list(s):
            if s is None:
                return None
            try:
                return [float(x) for x in s]
            except Exception:
                return None

        train = _to_float_list(train)
        val = _to_float_list(val)

        # Epochs
        epochs = stats.get("epochs")
        if isinstance(epochs, int) and (train or val):
            epochs = list(range(1, 1 + max(len(train or []), len(val or []))))
        elif isinstance(epochs, (list, tuple)):
            try:
                epochs = [int(x) for x in epochs]
            except Exception:
                epochs = None
        elif train:
            epochs = list(range(1, 1 + len(train)))
        elif val:
            epochs = list(range(1, 1 + len(val)))
        else:
            epochs = None

        # Counts
        counts = stats.get("labelled_counts") or stats.get("labelled") or stats.get("labeled_counts") or stats.get("labeled")
        if isinstance(counts, (list, tuple)):
            try:
                counts = [int(x) for x in counts]
            except Exception:
                counts = None
        else:
            counts = None

        return train, val, epochs, counts

    def _update_training_chart(self, stats: dict):
        """Draw training/validation loss and labelled image counts into the chart."""
        if not hasattr(self, "_training_ax"):
            return

        ax = self._training_ax
        ax.clear()

        train, val, epochs, counts = self._extract_training_data(stats)

        # Plot losses
        if train or val:
            if epochs is None:
                epochs = list(range(1, 1 + max(len(train or []), len(val or []))))
            if train:
                ax.plot(epochs[: len(train)], train, label="train loss", color="#1f77b4")
            if val:
                ax.plot(epochs[: len(val)], val, label="val loss", color="#ff7f0e")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper right", fontsize="small")

        # Plot counts on twin axis
        if counts:
            try:
                ax2 = ax.twinx()
                x = epochs[: len(counts)] if epochs else list(range(1, 1 + len(counts)))
                ax2.plot(x, counts, label="labelled images", color="#2ca02c", linestyle="--")
                ax2.set_ylabel("Labelled images")
                ax2.legend(loc="upper left", fontsize="small")
            except Exception:
                logging.exception("Failed to plot labelled counts")

        # Redraw canvas
        if hasattr(self, "_training_fig"):
            self._training_fig.tight_layout()
            canvas = getattr(self, "_training_canvas", None)
            if canvas and hasattr(canvas, "draw"):
                try:
                    canvas.draw()
                except Exception:
                    logging.exception("Failed to redraw training canvas")

    def _filter_objects_for_display(self, objects, min_confidence=0.6, max_objects=3, min_area_ratio=0.01):
        """Filter objects for display: min confidence threshold + top N by confidence/area + min area.
        
        Args:
            objects: List of detection dicts with 'confidence', 'bbox', optionally 'det_w', 'det_h'
            min_confidence: Minimum confidence threshold (default: 0.6)
            max_objects: Maximum number of objects to display (default: 3)
            min_area_ratio: Minimum area as ratio of image (default: 0.01 = 1% of image)
        
        Returns:
            Filtered list of objects, sorted by confidence descending
        """
        if not objects:
            return []

        # Filter by minimum confidence
        filtered = [obj for obj in objects if isinstance(obj, dict) and obj.get("confidence", 0.0) >= min_confidence]

        if not filtered:
            return []

        # Calculate area for each object and filter by minimum area
        for obj in filtered:
            bbox = obj.get("bbox")
            det_w = obj.get("det_w")
            det_h = obj.get("det_h")

            if bbox and det_w and det_h and len(bbox) == 4:
                # Bbox is [x1, y1, x2, y2] in absolute pixels
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                image_area = det_w * det_h
                obj["_area"] = area
                obj["_area_ratio"] = area / image_area if image_area > 0 else 0.0
            elif bbox and len(bbox) == 4:
                # Normalized [x, y, w, h]
                x, y, w, h = bbox
                obj["_area"] = w * h
                obj["_area_ratio"] = w * h  # Already normalized
            else:
                obj["_area"] = 0.0
                obj["_area_ratio"] = 0.0

        # Filter by minimum area ratio (exclude tiny objects)
        filtered = [obj for obj in filtered if obj.get("_area_ratio", 0.0) >= min_area_ratio]

        if not filtered:
            return []

        # Sort by confidence descending, then by area descending
        filtered.sort(key=lambda o: (o.get("confidence", 0.0), o.get("_area", 0.0)), reverse=True)

        # Return top N
        return filtered[:max_objects]

    def _paint_bboxes(self, pixmap, objects, offset_x=0, offset_y=0, thumb_w=None, thumb_h=None):
        """Paint bounding boxes and labels on the pixmap.
        
        Uses the same approach as paint_bboxes in view_helpers.py - draws directly on pixmap.

        Args:
            pixmap: QPixmap to paint on (square canvas)
            objects: List of detection dicts with keys 'class', 'confidence', 'bbox'
                     bbox format: [x1, y1, x2, y2] in absolute pixels (corners format from YOLOv8)
            offset_x: X offset of actual image within square canvas (letterbox) in logical pixels
            offset_y: Y offset of actual image within square canvas (letterbox) in logical pixels
            thumb_w: Actual width of thumbnail image (before letterboxing) in logical pixels
            thumb_h: Actual height of thumbnail image (before letterboxing) in logical pixels
        """
        if not pixmap or pixmap.isNull() or not objects:
            return pixmap

        try:
            from PySide6.QtGui import QPainter

            from .view_helpers import paint_bboxes

            # Get DPR and calculate dimensions
            dpr = pixmap.devicePixelRatio() or 1.0

            # Convert pixmap to QImage first to get logical pixel dimensions
            from PySide6.QtGui import QPixmap
            image = pixmap.toImage()
            if image.isNull():
                return pixmap

            # QImage is always in logical pixels (DPR removed)
            image_w_logical = image.width()
            image_h_logical = image.height()

            # Calculate image dimensions (use logical pixels)
            # thumb_w/thumb_h are in retina pixels, convert to logical
            # These represent the actual image size within the canvas (from PIL thumbnail metadata)
            if thumb_w:
                img_w = int(thumb_w / dpr) if dpr > 0 else thumb_w
            else:
                img_w = image_w_logical
            if thumb_h:
                img_h = int(thumb_h / dpr) if dpr > 0 else thumb_h
            else:
                img_h = image_h_logical
            offset_x_logical = int(offset_x / dpr) if dpr > 0 else offset_x
            offset_y_logical = int(offset_y / dpr) if dpr > 0 else offset_y

            logging.debug(f"[BBOX-FIX] _paint_bboxes: dpr={dpr}, thumb_w={thumb_w}, thumb_h={thumb_h}, img_w={img_w}, img_h={img_h}, offset=({offset_x_logical}, {offset_y_logical}), image_size=({image_w_logical}, {image_h_logical})")

            # CRITICAL: img_w/img_h must match the actual displayed image size in the QImage
            # The QImage canvas is image_w_logical x image_h_logical (typically 800x800 for thumb_size=800)
            # The image is positioned at (offset_x_logical, offset_y_logical) with size (img_w, img_h)
            # Detection coordinates are in (det_w, det_h) space, so we scale by (img_w/det_w, img_h/det_h)
            # Then add offset to position in canvas coordinates

            # Draw directly on QImage (already converted above)
            painter = QPainter(image)
            if not painter.isActive():
                logging.error("[BBOX-DEBUG] Painter failed to initialize!")
                return pixmap

            # Draw bboxes on QImage (logical coordinates)
            paint_bboxes(painter, objects, offset_x_logical, offset_y_logical, img_w, img_h)
            painter.end()

            # Convert back to QPixmap with DPR=1.0
            result_pixmap = QPixmap.fromImage(image)
            result_pixmap.setDevicePixelRatio(1.0)

            return result_pixmap
        except Exception as e:
            logging.exception(f"Failed to paint bboxes: {e}")
            # Return original pixmap on error - don't fail entire thumbnail rendering
            return pixmap

    def _refresh_thumbnail_badges(self):
        """Refresh label badges and bboxes on all thumbnail labels."""
        # Debounce badge refresh to avoid excessive repaints
        if not hasattr(self, "_badge_refresh_timer"):
            from PySide6.QtCore import QTimer
            self._badge_refresh_timer = QTimer(self)  # Parent to self to ensure it stays alive
            self._badge_refresh_timer.setSingleShot(True)
            self._badge_refresh_timer.timeout.connect(self._do_refresh_thumbnail_badges)
        
        # Debounce: restart timer instead of immediate refresh
        self._badge_refresh_timer.stop()
        self._badge_refresh_timer.start(100)  # 100ms debounce
    
    def _do_refresh_thumbnail_badges(self):
        """Actually perform badge refresh (called after debounce)."""
        # Don't purge caches on every refresh - only clear in-memory pixmap caches if needed
        # Detection cache file should persist across refreshes

        import time
        t0 = time.perf_counter()
        try:
            # Repaint all thumbnails to show updated predictions
            # OPTIMIZATION: Cache state access to avoid repeated getattr calls
            # Use getattr with default instead of hasattr (faster)
            state = getattr(self, "_last_browser_state", None)
            detected_objects = getattr(state, "detected_objects", {}) if state else {}
            group_info_dict = getattr(state, "group_info", {}) if state else {}
            
            # State data is available (logged at debug level if needed)
            
            # OPTIMIZATION: Batch visibility changes to avoid expensive show() calls (2.6s -> target <1s)
            # Track which widgets need to be shown/hidden, apply at end in one batch
            widgets_to_show = set()
            widgets_to_hide = set()

            repaint_count = 0
            with_prob = 0
            with_label = 0
            primary_prob = None
            primary_label = None
            primary_name = None
            # OPTIMIZATION: Use getattr instead of hasattr (faster)
            selection_model = getattr(self.viewmodel, "selection_model", None)
            primary = selection_model.primary() if selection_model else None
            if primary:
                try:
                    primary_name = os.path.basename(primary)
                except Exception:
                    primary_name = None
            # OPTIMIZATION: Only refresh visible thumbnails (80-90% reduction in work)
            # Get viewport visible region to filter labels
            visible_labels = []
            if hasattr(self, 'scroll_area') and self.scroll_area and hasattr(self.scroll_area, 'viewport'):
                try:
                    from PySide6.QtCore import QRect
                    viewport = self.scroll_area.viewport()
                    grid_widget = self.scroll_area.widget()
                    if grid_widget:
                        # Get visible region in viewport coordinates
                        visible_rect = viewport.rect()
                        # Map to grid widget coordinates
                        top_left = grid_widget.mapFrom(self.scroll_area.viewport(), visible_rect.topLeft())
                        bottom_right = grid_widget.mapFrom(self.scroll_area.viewport(), visible_rect.bottomRight())
                        
                        # Filter labels that intersect visible region
                        for (row, col), label in self.label_refs.items():
                            if label.isVisible():
                                label_rect = label.geometry()
                                # Check intersection (with margin for partial visibility)
                                margin = 50  # Include labels slightly outside viewport
                                if (label_rect.bottom() + margin >= top_left.y() and 
                                    label_rect.top() - margin <= bottom_right.y() and
                                    label_rect.right() + margin >= top_left.x() and 
                                    label_rect.left() - margin <= bottom_right.x()):
                                    visible_labels.append(((row, col), label))
                    else:
                        # Fallback: use all visible labels
                        visible_labels = [((r, c), lbl) for (r, c), lbl in self.label_refs.items() if lbl.isVisible()]
                except Exception:
                    # Fallback on error: use all visible labels
                    visible_labels = [((r, c), lbl) for (r, c), lbl in self.label_refs.items() if lbl.isVisible()]
            else:
                # Fallback: use all labels if no scroll area
                visible_labels = list(self.label_refs.items())
            
            # OPTIMIZATION: Process group badges for ALL labels (not just visible) when group_info is present
            # This ensures group badges appear even when scrolling to new thumbnails
            # Other badge updates (predictions, labels) are limited to visible labels for performance
            group_info_dict = getattr(state, "group_info", {}) if state else {}
            process_all_for_groups = bool(group_info_dict)
            
            # Process group badges for all labels if group_info is present
            if process_all_for_groups:
                all_labels = list(self.label_refs.items())
                # Removed frequent logging.info - called on every badge refresh (performance optimization)
                # logging.info(f"[badge-refresh] Processing ALL {len(all_labels)} labels for group badges (group_info present)")
            else:
                all_labels = visible_labels
                # logging.info(f"[badge-refresh] Processing {len(visible_labels)} visible labels out of {len(self.label_refs)} total")
            
            # Initialize counters for debugging
            badges_updated = 0
            bboxes_updated = 0
            groups_updated = 0
            
            for (row, col), label in all_labels:
                try:
                    # OPTIMIZATION: Direct attribute access (faster than hasattr + attribute)
                    # _thumb_filename is always set, so use direct access with try/except
                    try:
                        fname = label._thumb_filename
                    except AttributeError:
                        fname = None
                    if not fname:
                        raise ValueError(f"Label at ({row}, {col}) missing _thumb_filename attribute")

                    # Get full path from model for state lookup
                    # OPTIMIZATION: Cache path lookups to reduce repeated model calls
                    if not hasattr(self, "_fname_to_path_cache"):
                        self._fname_to_path_cache = {}
                    label_path = self._fname_to_path_cache.get(fname)
                    if not label_path:
                        label_path = self.viewmodel.model.get_image_path(fname)
                        if label_path:
                            self._fname_to_path_cache[fname] = label_path
                    if not label_path:
                        raise ValueError(f"Model returned no path for filename {fname}")

                    # OPTIMIZATION: Cache state lookups to reduce dict.get calls
                    current_label = self.viewmodel.model.get_state(label_path)
                    current_prob = None
                    if state:
                        # OPTIMIZATION: Direct dict access with try/except (faster than .get() for hot paths)
                        # predicted_probabilities is a dict, so use direct access
                        try:
                            current_prob = state.predicted_probabilities[fname]
                        except KeyError:
                            current_prob = None
                    if current_prob is not None:
                        with_prob += 1
                    if current_label:
                        with_label += 1
                    if primary_name and fname == primary_name:
                        primary_prob = current_prob
                        primary_label = current_label

                    # Determine current objects list
                    # OPTIMIZATION: Direct dict access with try/except (faster than .get() for hot paths)
                    try:
                        objects = detected_objects[fname] or []
                    except KeyError:
                        objects = []
                    # Filter objects for display (same as debug_cell)
                    if objects:
                        objects = self._filter_objects_for_display(objects, min_confidence=0.6, max_objects=3)

                    # Check if badge needs updating (prediction or label changed)
                    last_prob = getattr(label, "_last_prediction_prob", None)
                    last_label = getattr(label, "_last_label_state", None)
                    last_objects_count = getattr(label, "_last_objects_count", 0)

                    # Skip repaint if nothing changed (optimization to reduce CPU)
                    # Handle None comparisons correctly: None != None is False, but we want to detect when prob appears/disappears
                    prob_changed = False
                    if current_prob is not None and last_prob is None:
                        prob_changed = True  # Prediction appeared
                    elif current_prob is None and last_prob is not None:
                        prob_changed = True  # Prediction disappeared
                    elif current_prob is not None and last_prob is not None:
                        prob_changed = abs(current_prob - last_prob) > 0.001  # Prediction changed significantly

                    label_changed = current_label != last_label
                    objects_changed = len(objects) != last_objects_count

                    # PERFORMANCE: Use getattr with default instead of hasattr+getattr to reduce calls
                    # Always repaint on first load (when _last_prediction_prob doesn't exist yet)
                    # OR when we have a prediction but haven't displayed it yet (force initial display)
                    is_first_load = getattr(label, "_last_prediction_prob", None) is None
                    # Force repaint if we have a prediction but haven't cached it yet (first time seeing this prediction)
                    needs_initial_display = current_prob is not None and getattr(label, "_last_prediction_prob", None) is None
                    # Also force repaint if prediction exists but badge hasn't been painted yet (tracked via _badge_painted flag)
                    # Reset _badge_painted if prediction changed (need to repaint with new prediction)
                    badge_painted = getattr(label, "_badge_painted", False)
                    if badge_painted is False:  # First time or explicitly False
                        label._badge_painted = False  # type: ignore[attr-defined]
                    elif prob_changed:
                        label._badge_painted = False  # type: ignore[attr-defined]  # Reset if prediction changed
                    else:
                        label._badge_painted = badge_painted  # type: ignore[attr-defined]  # Keep existing value
                    # Force repaint if we have a prediction but badge wasn't painted yet
                    # OR if we have a prediction and it's the first time we're seeing it (even if badge was painted before)
                    needs_badge_paint = (current_prob is not None and not getattr(label, "_badge_painted", False)) or (current_prob is not None and is_first_load)

                    # Always update group badges (independent of prediction changes)
                    # Update group badge overlay BEFORE the continue check
                    group_badge_overlay = getattr(label, "_group_badge_overlay", None)
                    if group_badge_overlay and state:
                        # Use group_info_dict from above (already extracted)
                        if group_info_dict:
                            group_info = group_info_dict.get(fname, {})
                            if group_info:
                                is_best = group_info.get("is_group_best", False)
                                group_size = group_info.get("group_size", 1)
                                group_id = group_info.get("group_id")

                                # Always show badge if we have group info (to show group_id)
                                if group_id is not None or is_best or group_size > 1:
                                    # Removed frequent logging.info - called for every image with group info (performance optimization)
                                    # logging.info(f"[group-badge] Showing badge for {fname}: is_best={is_best}, group_size={group_size}, group_id={group_id}")
                                    group_badge_overlay.set_group_info(is_best=is_best, group_size=group_size, group_id=group_id)
                                    label_w = label.width()
                                    label_h = label.height()
                                    group_badge_overlay.setGeometry(0, 0, label_w, label_h)
                                    # Batch visibility change (don't call show() directly)
                                    widgets_to_show.add(group_badge_overlay)
                                    widgets_to_hide.discard(group_badge_overlay)  # Remove from hide set if it was there
                                    group_badge_overlay.raise_()
                                    groups_updated += 1
                                else:
                                    widgets_to_hide.add(group_badge_overlay)
                                    widgets_to_show.discard(group_badge_overlay)
                            else:
                                # No group info for this file - hide badge
                                widgets_to_hide.add(group_badge_overlay)
                                widgets_to_show.discard(group_badge_overlay)
                        else:
                            # No group_info_dict in state - hide badge
                            widgets_to_hide.add(group_badge_overlay)
                            widgets_to_show.discard(group_badge_overlay)
                    elif group_badge_overlay:
                        # No state available - hide badge
                        widgets_to_hide.add(group_badge_overlay)
                        widgets_to_show.discard(group_badge_overlay)

                    # Skip pixmap repaint if nothing changed (but still update overlays if needed)
                    needs_pixmap_repaint = (prob_changed or label_changed or objects_changed or is_first_load or 
                                           needs_initial_display or needs_badge_paint)
                    
                    # Always update badge/bbox overlays (widgets) if we have a label or prediction
                    # This ensures badges appear even when processing all labels for groups
                    # Update badge overlay widget for labeled images, or predicted badge for unlabeled
                    # Verify filename matches - fail fast on mismatch
                    label_filename = getattr(label, "_thumb_filename", None)
                    if label_filename != fname:
                        raise ValueError(f"Filename mismatch in _refresh_thumbnail_badges badge: label._thumb_filename={label_filename}, state fname={fname}")

                    # ALWAYS update bbox overlay if objects exist (independent of pixmap repaint)
                    if objects:
                        thumb_w = getattr(label, "_thumb_width", self.thumb_size)
                        thumb_h = getattr(label, "_thumb_height", self.thumb_size)
                        offset_x = getattr(label, "_thumb_offset_x", 0)
                        offset_y = getattr(label, "_thumb_offset_y", 0)
                        # Update bounding box overlay widget (reusable component)
                        # Only show if objects have bbox data
                        has_bboxes = any(obj.get("bbox") is not None for obj in objects)
                        bbox_overlay = getattr(label, "_bbox_overlay", None)
                        # Verify filename matches to prevent overlay mismatch - fail fast on mismatch
                        if label_filename != fname:
                            raise ValueError(f"Filename mismatch in _refresh_thumbnail_badges: label._thumb_filename={label_filename}, state fname={fname}")

                        if bbox_overlay and has_bboxes:
                            # Get full path from model using filename - fail fast if not found
                            if not fname:
                                raise ValueError("fname is None when updating bbox overlay in _refresh_thumbnail_badges")
                            if label_filename != fname:
                                raise ValueError(f"Filename mismatch: label._thumb_filename={label_filename}, state fname={fname}")

                            label_path = self.viewmodel.model.get_image_path(fname)
                            if not label_path:
                                raise ValueError(f"Model returned no path for filename {fname}")

                            try:
                                from PIL import Image
                                # OPTIMIZATION: Get original size from thumbnail metadata to avoid redundant Image.open
                                # Thumbnails store original size in their metadata
                                orig_w, orig_h = None, None
                                thumb = label.pixmap()
                                if thumb:
                                    # Try to get size from thumbnail metadata (stored during thumbnail creation)
                                    # Check if we cached it on the label
                                    cached_size = getattr(label, "_cached_image_size", None)
                                    if cached_size:
                                        orig_w, orig_h = cached_size
                                    else:
                                        # Fallback: open image to get size (only if not cached)
                                        with Image.open(label_path) as img:
                                            orig_w, orig_h = img.size
                                            # Cache for future use
                                            label._cached_image_size = (orig_w, orig_h)  # type: ignore[attr-defined]
                                
                                if orig_w and orig_h:
                                    bbox_overlay.set_detections(objects, original_image_size=(orig_w, orig_h))
                                else:
                                    # Last resort: open image
                                    with Image.open(label_path) as img:
                                        bbox_overlay.set_detections(objects, original_image_size=img.size)
                                # With setScaledContents(True), pixmap fills entire label
                                # So overlay should match label size, not thumbnail dimensions
                                label_w = label.width()
                                label_h = label.height()
                                bbox_overlay.setGeometry(0, 0, label_w, label_h)
                                # Batch visibility change - always add to show set
                                widgets_to_show.add(bbox_overlay)
                                widgets_to_hide.discard(bbox_overlay)
                                bbox_overlay.raise_()
                                bboxes_updated += 1
                                logging.debug(f"[BBOX-WIDGET] Updated overlay for {fname}: {len(objects)} objects with bboxes")
                            except Exception as e:
                                logging.exception(f"[BBOX-WIDGET] Failed to update overlay for {fname}: {e}")
                        elif bbox_overlay:
                            # Hide if no bboxes
                            widgets_to_hide.add(bbox_overlay)
                            widgets_to_show.discard(bbox_overlay)
                            bbox_overlay.set_detections([], original_image_size=None)
                            if objects:
                                logging.info(f"[BBOX-WIDGET] Hiding overlay for {fname}: {len(objects)} objects but no bbox data")

                    badge_overlay = getattr(label, "_badge_overlay", None)
                    if badge_overlay:
                        # Always set badge data, even if empty, to ensure widget state is correct
                        if current_label:
                            is_auto = self._is_auto_labeled(fname)
                            source = "auto" if is_auto else "manual"
                            # Always show probability score, even for manual labels (but skip if NaN)
                            prob = current_prob if current_prob is not None and current_prob == current_prob else None
                            badge_overlay.set_badge(current_label, source, prob)
                            # Position overlay to match label size (top-left corner)
                            label_w = label.width() or self.thumb_size
                            label_h = label.height() or self.thumb_size
                            badge_overlay.setGeometry(0, 0, label_w, label_h)
                            # Batch visibility change - always add to show set
                            widgets_to_show.add(badge_overlay)
                            widgets_to_hide.discard(badge_overlay)
                            badge_overlay.raise_()
                            badges_updated += 1
                        elif current_prob is not None and current_prob == current_prob:  # not None and not NaN
                            # Show predicted label for unlabeled images if we have a probability
                            predicted_label = "keep" if current_prob > 0.5 else "trash"
                            badge_overlay.set_badge(predicted_label, "predicted", current_prob)
                            label_w = label.width()
                            label_h = label.height()
                            badge_overlay.setGeometry(0, 0, label_w, label_h)
                            # Batch visibility change - always add to show set
                            widgets_to_show.add(badge_overlay)
                            widgets_to_hide.discard(badge_overlay)
                            badge_overlay.raise_()
                            badges_updated += 1
                        else:
                            widgets_to_hide.add(badge_overlay)
                            widgets_to_show.discard(badge_overlay)
                    
                    # Only repaint pixmap if something actually changed (not just for group processing)
                    if needs_pixmap_repaint:
                        # PERFORMANCE: Use getattr with default (already optimized, no change needed)
                        # Use base_pixmap if available (loaded thumbnail), else use current pixmap
                        source_pixmap = getattr(label, "base_pixmap", None) or label.pixmap()

                        if source_pixmap and not source_pixmap.isNull():
                            pixmap_copy = source_pixmap.copy()

                        # Group badge overlay is updated above (before the continue check)
                        # This ensures badges are always updated even if predictions haven't changed

                        # Paint rating stars

                        # Re-apply to label
                        label.setPixmap(pixmap_copy)
                        # Cache current state to avoid unnecessary repaints
                        label._last_prediction_prob = current_prob
                        label._last_label_state = current_label
                        label._last_objects_count = len(objects)
                        label._badge_painted = True  # type: ignore[attr-defined]
                        repaint_count += 1
                except Exception:
                    logging.exception(f"Failed to refresh thumbnail badge for label at ({row}, {col})")

            # OPTIMIZATION: Apply all visibility changes in one batch to reduce Qt overhead
            # Hide widgets first (faster), then show (triggers repaint)
            # OPTIMIZATION: Only process widgets that actually need state change
            widgets_to_hide_filtered = {w for w in widgets_to_hide if w.isVisible()}
            widgets_to_show_filtered = {w for w in widgets_to_show if not w.isVisible()}
            
            for widget in widgets_to_hide_filtered:
                widget.hide()
            for widget in widgets_to_show_filtered:
                # Verify widget has valid geometry before showing
                parent = widget.parent()
                if not parent:
                    logging.warning(f"[badge-refresh] Widget {type(widget).__name__} has no parent, skipping")
                    continue
                # Ensure parent is visible first
                if not parent.isVisible():
                    logging.warning(f"[badge-refresh] Parent of {type(widget).__name__} is not visible, showing parent")
                    parent.show()
                if widget.width() <= 0 or widget.height() <= 0:
                    if parent.width() > 0 and parent.height() > 0:
                        widget.setGeometry(0, 0, parent.width(), parent.height())
                        logging.info(f"[badge-refresh] Fixed geometry for {type(widget).__name__}: {widget.width()}x{widget.height()} (parent: {parent.width()}x{parent.height()})")
                # Always show widget (don't check isVisible() - force show)
                widget.show()
                widget.raise_()  # Ensure widget is on top after showing
                # Also ensure parent is updated and raised BEFORE widget update
                parent.update()
                parent.raise_()  # Ensure parent is on top
                # Force widget repaint AFTER parent is updated
                widget.update()  # Schedule async repaint
                widget.repaint()  # Force synchronous repaint (blocks until painted)
                # Widget should now be visible and painted
            
            t1 = time.perf_counter()
            # Removed frequent logging.info - called on every badge refresh (performance optimization)
            # if repaint_count > 0 or widgets_to_show or widgets_to_hide:
            #     logging.info(f"[badge-refresh] Repainted {repaint_count}/{len(self.label_refs)} thumbnails, showing {len(widgets_to_show)} widgets (badges: {badges_updated}, bboxes: {bboxes_updated}, groups: {groups_updated}), hiding {len(widgets_to_hide)} widgets")
        except Exception:
            logging.exception("Failed in _refresh_thumbnail_badges")
            raise  # Fail fast - badge refresh errors should be visible

    def _set_label_pixmap(self, label, pixmap, ctx=""):
        # Create high-DPI pixmap for crisp overlays on retina displays.
        # Paint overlays at full resolution, then set device pixel ratio.
        try:
            if pixmap is None:
                label.clear()
                return

            # Get device pixel ratio for retina displays
            dpr = self._dpr
            target_size = int(self.thumb_size * dpr)

            # Pixmap should already be at retina resolution from _convert_to_pixmap
            # Just verify and ensure DPR is set
            try:
                if isinstance(pixmap, QPixmap) and not pixmap.isNull():
                    if pixmap.devicePixelRatio() != dpr:
                        pixmap.setDevicePixelRatio(dpr)
            except Exception:
                pass

            # Paint bboxes and label badge if available
            try:
                # Use stored filename - fail fast if not set
                fname = getattr(label, "_thumb_filename", None)
                if not fname:
                    raise ValueError("Label missing _thumb_filename attribute in _set_label_pixmap")

                if fname:

                    # Get actual thumbnail dimensions from label storage
                    thumb_w = getattr(label, "_thumb_width", self.thumb_size)
                    thumb_h = getattr(label, "_thumb_height", self.thumb_size)
                    offset_x = getattr(label, "_thumb_offset_x", 0)
                    offset_y = getattr(label, "_thumb_offset_y", 0)

                    # Get detected objects from state
                    state = getattr(self, "_last_browser_state", None)
                    detected_objects = getattr(state, "detected_objects", {}) if state else {}
                    objects = detected_objects.get(fname, [])

                    # Filter objects before painting: min confidence + top N by confidence/area
                    if objects:
                        filtered_objects = self._filter_objects_for_display(objects, min_confidence=0.6, max_objects=3)
                        if filtered_objects:
                            # CRITICAL: Ensure det_w/det_h are set for proper bbox scaling
                            # If missing (from old cache), infer from image file
                            label_path = label.toolTip()
                            if label_path:
                                for obj in filtered_objects:
                                    if not obj.get("det_w") or not obj.get("det_h"):
                                        try:
                                            from PIL import Image

                                            from src.object_detection import DetectionConfig
                                            img = Image.open(label_path)
                                            orig_w, orig_h = img.size
                                            # Detection resizes to max_size=800 while preserving aspect ratio
                                            max_size = DetectionConfig().max_size  # Default 800
                                            if max(orig_w, orig_h) > max_size:
                                                ratio = max_size / float(max(orig_w, orig_h))
                                                det_w = max(1, int(orig_w * ratio))
                                                det_h = max(1, int(orig_h * ratio))
                                            else:
                                                det_w, det_h = orig_w, orig_h
                                            obj["det_w"] = det_w
                                            obj["det_h"] = det_h
                                            logging.debug(f"[BBOX-FIX] Inferred det_w={det_w}, det_h={det_h} for {fname}")
                                        except Exception as e:
                                            logging.error(f"[BBOX-FIX] Failed to infer det_w/det_h for {fname}: {e}")
                                            raise  # Fail fast - bbox scaling requires det_w/det_h

                            # Update bounding box overlay widget (reusable component)
                            # Only show if objects have bbox data
                            has_bboxes = any(obj.get("bbox") is not None for obj in filtered_objects)
                            bbox_overlay = getattr(label, "_bbox_overlay", None)
                            # Removed frequent logging.info - called for every thumbnail with objects (performance optimization)
                            # logging.info(f"[BBOX-WIDGET] {fname}: overlay={bbox_overlay is not None}, filtered_objects={len(filtered_objects)}, has_bboxes={has_bboxes}")
                            if bbox_overlay and has_bboxes:
                                # Get full path from model using filename - fail fast if not found
                                if not fname:
                                    raise ValueError("fname is None when updating bbox overlay")
                                label_path = self.viewmodel.model.get_image_path(fname)
                                if not label_path:
                                    raise ValueError(f"Model returned no path for filename {fname}")

                                try:
                                    from PIL import Image
                                    img = Image.open(label_path)
                                    orig_w, orig_h = img.size
                                    bbox_overlay.set_detections(filtered_objects, original_image_size=(orig_w, orig_h))
                                    # With setScaledContents(True), pixmap fills entire label
                                    # So overlay should match label size, not thumbnail dimensions
                                    label_w = label.width()
                                    label_h = label.height()
                                    bbox_overlay.setGeometry(0, 0, label_w, label_h)
                                    # OPTIMIZATION: Batch visibility change (don't call show() directly)
                                    # Visibility will be handled by badge refresh batching
                                    bbox_overlay.raise_()
                                    # Removed frequent logging.info - called for every thumbnail update (performance optimization)
                                    # logging.info(f"[BBOX-WIDGET] Updated overlay for {fname}: {len(filtered_objects)} objects with bboxes, label={label_w}x{label_h}, widget visible={bbox_overlay.isVisible()}")
                                except Exception as e:
                                    logging.exception(f"[BBOX-WIDGET] Failed to update overlay for {fname}: {e}")
                            elif bbox_overlay:
                                # Hide if no bboxes
                                bbox_overlay.hide()
                                # Removed frequent logging.info - called for every thumbnail without bbox data (performance optimization)
                                # if filtered_objects:
                                #     logging.info(f"[BBOX-WIDGET] Hiding overlay for {fname}: {len(filtered_objects)} objects but no bbox data")

                            # Offsets and thumb dimensions are already at retina resolution (from pixmap sampling)
                            # Do NOT scale again - they're already correct for the retina pixmap
                            # Note: bboxes now drawn via widget overlay, but keep pixmap painting as fallback
                            # pixmap = self._paint_bboxes(pixmap, filtered_objects, offset_x, offset_y, thumb_w, thumb_h)
                    else:
                        # Removed frequent logging.debug - called for every thumbnail without objects (performance optimization)
                        # logging.debug(f"[OVERLAY-TEST] No objects to paint for {fname}")
                        pass

                    # Update badge overlay widget if this label has a state OR if we have a prediction for unlabeled images
                    # Verify filename matches - fail fast on mismatch
                    label_filename = getattr(label, "_thumb_filename", None)
                    if label_filename != fname:
                        raise ValueError(f"Filename mismatch in _set_label_pixmap badge: label._thumb_filename={label_filename}, state fname={fname}")

                    # Get full path from model for state lookup - fail fast if not found
                    if not fname:
                        raise ValueError("fname is None in _set_label_pixmap")
                    label_path = self.viewmodel.model.get_image_path(fname)
                    if not label_path:
                        raise ValueError(f"Model returned no path for filename {fname}")
                    current_label = self.viewmodel.model.get_state(label_path)
                    prob = state.predicted_probabilities.get(fname) if state else None
                    if prob is not None and prob != prob:  # NaN check
                        prob = None

                    badge_overlay = getattr(label, "_badge_overlay", None)
                    badge_was_painted = False
                    if badge_overlay and label_filename == fname:
                        if current_label:
                            is_auto = self._is_auto_labeled(fname)
                            source = "auto" if is_auto else "manual"
                            badge_overlay.set_badge(current_label, source, prob)
                            label_w = label.width()
                            label_h = label.height()
                            badge_overlay.setGeometry(0, 0, label_w, label_h)
                            # OPTIMIZATION: Don't call show() directly - let badge refresh handle visibility
                            badge_overlay.raise_()
                            label._last_label_state = current_label
                            label._last_prediction_prob = prob
                            label._last_objects_count = len(objects)
                            badge_was_painted = True
                        elif prob is not None and prob == prob:  # not None and not NaN - unlabeled image with prediction
                            # Show predicted label for unlabeled images
                            predicted_label = "keep" if prob > 0.5 else "trash"
                            badge_overlay.set_badge(predicted_label, "predicted", prob)
                            label_w = label.width()
                            label_h = label.height()
                            badge_overlay.setGeometry(0, 0, label_w, label_h)
                            # OPTIMIZATION: Don't call show() directly - let badge refresh handle visibility
                            badge_overlay.raise_()
                            label._last_label_state = None
                            label._last_prediction_prob = prob
                            label._last_objects_count = len(objects)
                            badge_was_painted = True
                        else:
                            badge_overlay.hide()

                    # Only mark badge as painted if we actually painted one
                    if badge_was_painted:
                        label._badge_painted = True  # type: ignore[attr-defined]
                    else:
                        # No badge painted yet (no label, no prediction) - ensure flag is False
                        # Use getattr instead of hasattr (faster)
                        if not getattr(label, "_badge_painted", False):
                            label._badge_painted = False  # type: ignore[attr-defined]

                    # Paint rating stars
            except Exception:
                logging.exception("Failed to apply overlays in _set_label_pixmap")

            # CRITICAL: Create one final deep copy before setting pixmap to prevent Qt from sharing
            # Even though we already created a copy, Qt's setPixmap might optimize and share internally
            # Converting to QImage and back forces a complete, independent copy
            try:
                final_img = pixmap.toImage()
                if final_img.isNull():
                    logging.error(f"[THUMBNAIL] Failed to convert pixmap to image for final copy (fname={getattr(label, '_thumb_filename', 'unknown')})")
                    label.setPixmap(pixmap)  # Fallback to original
                else:
                    final_pixmap = QPixmap.fromImage(final_img)
                    if final_pixmap.isNull():
                        logging.error(f"[THUMBNAIL] Failed to create final pixmap copy (fname={getattr(label, '_thumb_filename', 'unknown')})")
                        label.setPixmap(pixmap)  # Fallback to original
                    else:
                        final_pixmap.setDevicePixelRatio(pixmap.devicePixelRatio())
                        # Store the filename on the pixmap to track which image it represents
                        # This helps debug if the same pixmap is being reused
                        fname = getattr(label, "_thumb_filename", None)
                        if fname:
                            # Store filename as a property (won't affect rendering)
                            final_pixmap._assigned_filename = fname  # type: ignore[attr-defined]
                        label.setPixmap(final_pixmap)
                        # Removed pixmap filename verification - false positives when pixmap created before filename set
                        # This check was causing frequent ERROR logs without actual issues
            except Exception as e:
                logging.error(f"[THUMBNAIL] Failed to create final pixmap copy: {e}, using original")
                label.setPixmap(pixmap)  # Fallback to original on error
        except Exception:
            logging.exception("Failed in _set_label_pixmap")

    def _on_selection_model_changed(self, selected_paths):
        # Update immediate lightweight state (highlights) and debounce heavy refreshes
        self.selected_filenames = set(selected_paths)
        self._update_all_highlights()
        # Debounce full badge refresh to coalesce rapid clicks
        try:
            self._selection_debounce_timer.start()
        except Exception:
            # Fallback: if timer not available, run immediately
            self._on_selection_debounced()

    def _on_selection_debounced(self):
        """Run heavier UI updates after selection events settle."""
        try:
            # EXIF display is updated via _on_exif_changed signal from viewmodel
            # when selection changes, viewmodel automatically loads details and emits exif_changed

            # Only refresh badges if browser state is present
            if getattr(self, "_last_browser_state", None):
                self._refresh_thumbnail_badges()
        except Exception:
            logging.exception("Failed in _on_selection_debounced")

    def _get_label_color(self, label: str | None) -> tuple[str, str]:
        """Get border color and width for label state. Returns (color, width)."""
        if label == "keep":
            return COLOR_KEEP, "3px"
        elif label == "trash":
            return COLOR_TRASH, "3px"
        else:
            return COLOR_UNLABELED, "2px"

    def _update_label_highlight(self, label, full_path):
        """Update label styling with color-coded borders based on label state."""
        current_label = self.viewmodel.model.get_state(full_path) if full_path else None
        is_selected = full_path in getattr(self, "selected_filenames", set())

        border_color, border_width = self._get_label_color(current_label)
        if is_selected:
            border_color = "#4a5a6a"  # Lightroom palette for selected
            border_width = "2px"
        
        styles = [
            f"border: {border_width} solid {border_color};",
            "border-radius: 2px;",
        ]
        if is_selected:
            styles.append("background: #3a4a5a;")  # Lightroom palette
        # Note: Qt stylesheets don't support box-shadow, removed to avoid warnings

        # OPTIMIZATION: Cache stylesheet to avoid redundant setStyleSheet calls
        # Store last style on label object to avoid expensive styleSheet() getter
        new_style = " ".join(styles)
        last_style = getattr(label, "_last_style", None)
        if last_style != new_style:
            label.setStyleSheet(new_style)
            label._last_style = new_style  # type: ignore[attr-defined]

    def _update_all_highlights(self):
        # OPTIMIZATION: Batch style updates to reduce Qt overhead (setStyleSheet is expensive)
        # Collect all style changes first, then apply in one pass
        style_updates = []
        for (row, col), label in self.label_refs.items():
            full_path = label.toolTip()
            current_label = self.viewmodel.model.get_state(full_path) if full_path else None
            is_selected = full_path in getattr(self, "selected_filenames", set())
            
            border_color, border_width = self._get_label_color(current_label)
            if is_selected:
                border_color = "#4a5a6a"  # Lightroom palette for selected
                border_width = "2px"
            
            styles = [
                f"border: {border_width} solid {border_color};",
                "border-radius: 2px;",
            ]
            if is_selected:
                # Use Lightroom palette for selected background
                styles.append("background: #3a4a5a;")
            
            new_style = " ".join(styles)
            current_style = label.styleSheet()
            if current_style != new_style:
                style_updates.append((label, new_style))
        
        # Apply all style updates in one batch (reduces Qt overhead)
        for label, style in style_updates:
            label.setStyleSheet(style)
            label._last_style = style  # type: ignore[attr-defined]  # Cache for next check
        
        # Update status bar to reflect selection count
        self._update_status_bar()
        # Show/hide batch toolbar based on selection count
        # OPTIMIZATION: selected_filenames is always set, use direct access
        selected_count = len(self.selected_filenames)
        if hasattr(self, "batch_toolbar"):
            self.batch_toolbar.setVisible(selected_count > 1)

    def _on_exif_changed(self, exif):
        """Handle EXIF data change from viewmodel signal."""
        primary = self.viewmodel.selection_model.primary()

        if not primary:
            self.exif_view.setText("No image selected.")
            try:
                if hasattr(self, "object_detection_placeholder"):
                    self.object_detection_placeholder.setText("No image selected.")
                if hasattr(self, "object_detection_list_layout"):
                    self._clear_layout(self.object_detection_list_layout)
            except Exception:
                logging.exception("Failed to clear object detection layout")
            return

        # Use EXIF data directly from signal (already loaded by viewmodel)
        path = self.viewmodel.model.get_image_path(primary)
        if not path:
            return

        lines = []
        fname = os.path.basename(primary)
        p = self._get_current_prediction_prob(fname)
        is_auto = self._is_auto_labeled(fname)

        # Get additional metadata
        label = self.viewmodel.model.get_state(path)

        # Format header
        lines.append(f"📷 {fname}\n")
        lines.append("=" * 50)

        # Label
        if label:
            label_icon = "✅" if label == "keep" else "❌"
            source_text = " (auto)" if is_auto else ""
            lines.append(f"\n🏷️  Label: {label_icon} {label.upper()}{source_text}")

        # Prediction info
        if p is not None and isinstance(p, (int, float)) and p == p:  # Check for NaN
            lines.append(f"\n🤖 Prediction: {round(p*100)}% Keep, {round((1-p)*100)}% Trash")
            # Calculate uncertainty score (same as used for sorting)
            uncertainty = 0.5 - abs(p - 0.5)
            uncertainty_pct = round(uncertainty * 100, 1)
            lines.append(f"   Uncertainty: {uncertainty_pct}% (higher = more uncertain)")
        elif p is None:
            # No prediction = highest uncertainty
            lines.append("\n🤖 Prediction: No prediction yet")
            lines.append("   Uncertainty: 100.0% (no prediction = highest uncertainty)")

        # EXIF data
        if exif:
            lines.append("\n📸 Camera Settings:")
            lines.append("-" * 50)

            # Common EXIF fields with better formatting
            exif_map = {
                "Make": "Camera",
                "Model": "Model",
                "DateTime": "Date Taken",
                "DateTimeOriginal": "Date Taken",
                "ExposureTime": "Shutter Speed",
                "FNumber": "Aperture",
                "ISOSpeedRatings": "ISO",
                "ISO": "ISO",
                "FocalLength": "Focal Length",
                "Flash": "Flash",
                "WhiteBalance": "White Balance",
            }

            for key, display_name in exif_map.items():
                if key in exif:
                    value = exif[key]
                    if key == "ExposureTime" and isinstance(value, (int, float)):
                        if value < 1:
                            value = f"1/{int(1/value)}s"
                        else:
                            value = f"{value}s"
                    elif key == "FNumber" and isinstance(value, (int, float)):
                        value = f"f/{value}"
                    elif key == "FocalLength" and isinstance(value, (int, float)):
                        value = f"{value}mm"
                    lines.append(f"  {display_name}: {value}")

            # Other EXIF data
            other_keys = [k for k in exif.keys() if k not in exif_map]
            if other_keys:
                lines.append("\n📋 Other Metadata:")
                lines.append("-" * 50)
                for key in sorted(other_keys)[:10]:
                    lines.append(f"  {key}: {exif[key]}")
        else:
            lines.append("\n📸 No EXIF data found.")
            # Show the fallback date that's actually being used
            try:
                from .grouping_service import extract_timestamp
                fallback_date = extract_timestamp(exif or {}, path)
                lines.append(f"   Date used (fallback): {fallback_date.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                pass  # Don't fail if date extraction fails

        # File info
        try:
            stat = os.stat(path)
            size_mb = stat.st_size / (1024 * 1024)
            lines.append(f"\n💾 File Size: {size_mb:.2f} MB")
        except Exception:
            pass

        # Image dimensions
        try:
            from PIL import Image
            with Image.open(path) as img:
                orig_w, orig_h = img.size
                lines.append(f"📐 Dimensions: {orig_w} x {orig_h} pixels")
        except Exception:
            pass

        self.exif_view.setText("\n".join(lines))

        # Update object detection display
        self._update_object_detection_display(primary)

    def _update_object_detection_display(self, primary_path: str):
        """Update the object detection display for the selected image."""
        if not primary_path:
            self.object_detection_placeholder.setText("No image selected.")
            self._clear_layout(self.object_detection_list_layout)
            return

        fname = os.path.basename(primary_path)

        # Get detected objects from the browser state
        detected_objects = getattr(self._state, "detected_objects", {}) if self._state else {}
        objects = detected_objects.get(fname, []) or []

        # Don't run synchronous detection here - it causes RecursionError in YOLOv8 model
        # Background detection task will handle missing detections automatically
        # If objects are missing bbox/det sizes, they'll be updated when background detection completes

        if not objects:
            self.object_detection_placeholder.setText("No objects detected in this image.")
            self._clear_layout(self.object_detection_list_layout)
            return

        # Populate chip-style list for detected objects
        self.object_detection_placeholder.setText("")
        # clear previous chips
        self._clear_layout(self.object_detection_list_layout)

        for obj in objects:
            # support full detection dicts ({'class','confidence','bbox'}) or tuple/name
            if isinstance(obj, dict):
                name = obj.get("class") or "object"
                score = obj.get("confidence")
            elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                name = str(obj[0])
                score = obj[1] if len(obj) > 1 else None
            else:
                name = str(obj)
                score = None

            chip = QLabel()
            if score is None:
                chip_text = str(name)
            else:
                try:
                    chip_text = f"{name} — {float(score) * 100:.0f}%"
                except Exception:
                    chip_text = f"{name} — {score}"

            chip.setText(chip_text)
            chip.setToolTip(str(name))
            chip.setStyleSheet("""
                QLabel {
                    background: #e3f2fd;
                    color: #1976d2;
                    padding: 6px 12px;
                    border-radius: 16px;
                    font-size: 9px;
                    font-weight: 500;
                    border: 1px solid #bbdefb;
                }
            """)
            chip.setFixedHeight(26)
            chip.setContentsMargins(0, 0, 0, 0)
            self.object_detection_list_layout.addWidget(chip)

    def _convert_to_pixmap(self, thumb) -> QPixmap:
        """Convert thumbnail (PIL Image or QPixmap) to QPixmap, preserving any PIL.Image.info metadata.
        
        Scales PIL images to retina resolution before conversion for crisp overlays.
        Uses caching to avoid repeated conversions of the same image.
        """
        if isinstance(thumb, QPixmap):
            return thumb

        # Get device pixel ratio for retina displays
        dpr = self._dpr
        target_size = int(self.thumb_size * dpr)

        # Generate cache key from PIL Image data (if available)
        # CRITICAL: Include filename in cache key to prevent same image data from different files sharing pixmaps
        cache_key = None
        if hasattr(thumb, "tobytes"):
            try:
                # Get filename from thumb.info if available (set during thumbnail creation)
                thumb_filename = None
                if hasattr(thumb, "info") and thumb.info:
                    # Try to get original path from metadata
                    thumb_filename = thumb.info.get("original_path") or thumb.info.get("thumb_filename")

                # Use image data + filename + DPR as cache key to ensure uniqueness per file
                thumb_bytes = thumb.tobytes()
                import hashlib
                cache_key_str = f"{thumb_bytes[:1024]}_{thumb_filename or 'unknown'}_{dpr}"
                cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
                # NOTE: Cache lookup disabled - we always create new pixmaps to prevent sharing
                # Even with unique cache keys, Qt might share pixmap data internally
                # Creating fresh pixmaps for each label ensures complete independence
            except Exception:
                pass

        # DON'T resize PIL image - it breaks coordinate mapping
        # The thumbnail is already at the correct size (128x128), and we'll scale it via DPR
        # Resizing here would change the coordinate space and break bbox alignment
        # Instead, let Qt handle scaling via setScaledContents(True) on the label

        # Use PIL.ImageQt for direct conversion, avoiding PNG round-trip that loses PIL.Image.info
        try:
            from PIL.ImageQt import ImageQt

            rgba_img = thumb.convert("RGBA") if hasattr(thumb, "convert") else thumb
            qimg = ImageQt(rgba_img)
            pm = QPixmap.fromImage(qimg)
            # Set device pixel ratio so Qt displays at correct size
            pm.setDevicePixelRatio(dpr)
            # Attach metadata to QPixmap for later retrieval
            if hasattr(thumb, "info") and thumb.info:
                logging.debug(f"[BBOX-DEBUG] Attaching PIL.Image.info to QPixmap: {thumb.info}")
                # Store as a temporary reference on the pixmap (won't survive pickling, but fine for GUI)
                pm._pil_info = thumb.info  # type: ignore[attr-defined]

            # DISABLED: Don't cache converted pixmaps to prevent sharing between labels
            # Each label needs a completely unique pixmap, even if the source image is the same
            # if cache_key:
            #     # Evict oldest entries if cache is full
            #     if len(self._converted_pixmap_cache) >= self._converted_pixmap_cache_max:
            #         # Remove oldest 25% of entries
            #         keys_to_remove = list(self._converted_pixmap_cache.keys())[:self._converted_pixmap_cache_max // 4]
            #         for k in keys_to_remove:
            #             del self._converted_pixmap_cache[k]
            #     self._converted_pixmap_cache[cache_key] = pm

            return pm
        except Exception as e:
            logging.debug(f"ImageQt conversion failed, falling back to PNG: {e}")

        # Fallback: PNG round-trip (loses metadata)
        try:
            from io import BytesIO

            buf = BytesIO()
            try:
                thumb.save(buf, format="PNG")
                data = buf.getvalue()
                pm = QPixmap()
                if pm.loadFromData(data):
                    # DISABLED: Don't cache converted pixmaps to prevent sharing between labels
                    # if cache_key:
                    #     if len(self._converted_pixmap_cache) >= self._converted_pixmap_cache_max:
                    #         keys_to_remove = list(self._converted_pixmap_cache.keys())[:self._converted_pixmap_cache_max // 4]
                    #         for k in keys_to_remove:
                    #             del self._converted_pixmap_cache[k]
                    #     self._converted_pixmap_cache[cache_key] = pm
                    return pm
            except Exception:
                pass
        except Exception:
            logging.exception("Failed to convert PIL image to QPixmap; returning empty pixmap")
        return QPixmap()

    def _on_thumbnail_loaded(self, path, thumb=None):
        try:
            # OPTIMIZATION: Cache basename (was called twice)
            path_filename = os.path.basename(path)
            # Log the actual path to track if same path is being loaded multiple times
            logging.debug(f"[THUMBNAIL] Loading thumbnail for path={path}, filename={path_filename}, has_thumb={thumb is not None}, thumb_id={id(thumb) if thumb else None}, _all_widgets_count={len(self._all_widgets)}")
            if not thumb:
                logging.warning(f"[THUMBNAIL] No thumbnail provided for {path_filename}")
                return

            # Use direct lookup in _all_widgets (maps filename -> label)
            label = self._all_widgets.get(path_filename)

            # If label not found, try multiple fallback strategies
            if not label:
                # OPTIMIZATION: Cache abspath results to avoid repeated calls
                path_abspath = None
                try:
                    path_abspath = os.path.abspath(str(path))
                except Exception:
                    pass
                
                # Strategy 1: Match by _thumb_filename attribute
                for candidate_label in self._all_widgets.values():
                    candidate_filename = getattr(candidate_label, "_thumb_filename", None)
                    if candidate_filename == path_filename:
                        label = candidate_label
                        logging.debug(f"[THUMBNAIL] Found label for {path_filename} via _thumb_filename match")
                        break

                # Strategy 2: Match by tooltip (full path)
                if not label and path_abspath:
                    for candidate_label in self._all_widgets.values():
                        tooltip_path = candidate_label.toolTip()
                        if tooltip_path:
                            try:
                                tooltip_abspath = os.path.abspath(str(tooltip_path))
                                if tooltip_abspath == path_abspath:
                                    label = candidate_label
                                    logging.debug(f"[THUMBNAIL] Found label for {path_filename} via tooltip match")
                                    break
                            except Exception:
                                if str(tooltip_path) == str(path):
                                    label = candidate_label
                                    logging.debug(f"[THUMBNAIL] Found label for {path_filename} via tooltip string match")
                                    break

            # Strategy 3 removed - too aggressive, could cause wrong assignments

            # If still not found, log detailed info and skip
            if not label:
                available = list(self._all_widgets.keys())[:20]  # First 20 for logging
                # Also check _thumb_filename attributes
                thumb_filenames = []
                visible_without_pixmap = []
                for candidate_label in list(self._all_widgets.values())[:20]:
                    fn = getattr(candidate_label, "_thumb_filename", None)
                    if fn:
                        thumb_filenames.append(fn)
                    if candidate_label.isVisible():
                        try:
                            pm = candidate_label.pixmap()
                            if not pm or pm.isNull():
                                visible_without_pixmap.append(fn or "no_filename")
                        except Exception:
                            visible_without_pixmap.append(fn or "no_filename")
                logging.warning(f"[THUMBNAIL] No label found for {path_filename} in _all_widgets (has {len(self._all_widgets)} entries). Available keys: {available[:10]}, _thumb_filenames: {thumb_filenames[:10]}, visible_without_pixmap: {visible_without_pixmap[:10]}")
                return

            # CRITICAL: Verify the stored filename matches BEFORE processing
            # This prevents assigning thumbnails to wrong labels
            label_filename = getattr(label, "_thumb_filename", None)
            if label_filename != path_filename:
                logging.error(f"[THUMBNAIL] FILENAME MISMATCH: Thumbnail for {path_filename} found label with _thumb_filename={label_filename}. Skipping to prevent wrong assignment.")
                return

            # OPTIMIZATION: Get grid position using O(1) reverse lookup instead of O(n) iteration
            grid_pos = self._label_to_grid_pos.get(label)

            # OPTIMIZATION: Combine pixmap checks to avoid redundant calls
            try:
                existing_pixmap = label.pixmap()
                if existing_pixmap and not existing_pixmap.isNull():
                    # Double-check filename still matches (race condition protection)
                    current_filename = getattr(label, "_thumb_filename", None)
                    if current_filename != path_filename:
                        logging.error(f"[THUMBNAIL] Label at {grid_pos} already displays {current_filename}, rejecting {path_filename} (race condition or mismatch)")
                        return
            except Exception:
                pass

            logging.debug(f"[THUMBNAIL] Assigning thumbnail {path_filename} to label with _thumb_filename={label_filename}, grid_pos={grid_pos}, label_id={id(label)}, label visible={label.isVisible()}")

            # KISS: Extract metadata and convert to pixmap - work in logical pixels, let Qt handle DPR
            # CRITICAL: Create unique copy of PIL Image before conversion to prevent sharing
            # If same PIL Image object is returned for different paths, this ensures each label gets unique data
            try:
                from PIL import Image
                if isinstance(thumb, Image.Image):
                    # Create a completely new PIL Image copy to prevent any sharing
                    thumb_copy = thumb.copy()
                    # Also copy the info dict to preserve metadata
                    if hasattr(thumb, "info") and thumb.info:
                        thumb_copy.info = dict(thumb.info)
                    thumb = thumb_copy
                    logging.debug(f"[THUMBNAIL] Created unique PIL Image copy for {path_filename}, thumb_id={id(thumb)}")
            except Exception as e:
                logging.warning(f"[THUMBNAIL] Failed to create PIL Image copy for {path_filename}: {e}")

            # Convert thumbnail to QPixmap first
            try:
                pixmap = self._convert_to_pixmap(thumb)
                logging.debug(f"[OVERLAY-TEST] Converted thumb to QPixmap: {pixmap.width()}x{pixmap.height()}, pixmap_id={id(pixmap)}")
            except Exception:
                logging.exception("Failed converting thumb to QPixmap")
                pixmap = QPixmap()

            # Extract metadata from PIL Image (stored at 1x logical resolution)
            # Default to canvas size, but try to get actual image size from PIL info
            thumb_width = self.thumb_size
            thumb_height = self.thumb_size
            offset_x = 0
            offset_y = 0
            try:
                if hasattr(thumb, "info") and thumb.info:
                    thumb_width = int(thumb.info.get("thumb_width", self.thumb_size) or self.thumb_size)
                    thumb_height = int(thumb.info.get("thumb_height", self.thumb_size) or self.thumb_size)
                    offset_x = int(thumb.info.get("thumb_offset_x", 0) or 0)
                    offset_y = int(thumb.info.get("thumb_offset_y", 0) or 0)
                # If info is missing, try to infer from PIL image dimensions
                # The PIL image is the canvas, but we can check if it's smaller than thumb_size
                elif hasattr(thumb, "size"):
                    # If PIL image is smaller than canvas, it's likely the actual image size
                    pil_w, pil_h = thumb.size
                    if pil_w < self.thumb_size or pil_h < self.thumb_size:
                        thumb_width = pil_w
                        thumb_height = pil_h
                        # Center on canvas
                        offset_x = (self.thumb_size - pil_w) // 2
                        offset_y = (self.thumb_size - pil_h) // 2
            except Exception as e:
                logging.exception(f"Failed to extract metadata from PIL.Image.info: {e}")

            # Scale metadata to match pixmap resolution (pixmap is already scaled by DPR in _convert_to_pixmap)
            if not pixmap.isNull():
                dpr = pixmap.devicePixelRatio() or 1.0
                # Store logical pixel dimensions first (before scaling)
                thumb_width_logical = thumb_width
                thumb_height_logical = thumb_height
                offset_x_logical = offset_x
                offset_y_logical = offset_y

                # Scale to retina for storage
                thumb_width = int(thumb_width * dpr)
                thumb_height = int(thumb_height * dpr)
                offset_x = int(offset_x * dpr)
                offset_y = int(offset_y * dpr)

                # Recompute offsets from dimensions in logical pixels, then scale to retina
                # pixmap.width() returns logical pixels (Qt divides by DPR automatically)
                canvas_size_logical = pixmap.width()
                offset_x_logical = (canvas_size_logical - thumb_width_logical) // 2
                offset_y_logical = (canvas_size_logical - thumb_height_logical) // 2
                # Scale recomputed offsets to retina
                offset_x = int(offset_x_logical * dpr)
                offset_y = int(offset_y_logical * dpr)

                # Final verification: ensure label filename still matches (prevent race conditions)
                label_filename_check = getattr(label, "_thumb_filename", None)
                if label_filename_check != path_filename:
                    # Filename changed during load - skip to prevent wrong image
                    return

                # Create a DEEP copy for base_pixmap to prevent sharing between labels
                # Convert to QImage and back to QPixmap to force a true deep copy
                try:
                    # Convert pixmap to image (forces deep copy of pixel data)
                    img = pixmap.toImage()
                    if img.isNull():
                        logging.error(f"[THUMBNAIL] Failed to convert pixmap to image for base_pixmap {path_filename}")
                        return
                    # Create new pixmap from image (truly independent copy)
                    base_pixmap_copy = QPixmap.fromImage(img)
                    if base_pixmap_copy.isNull():
                        logging.error(f"[THUMBNAIL] Failed to create pixmap from image for base_pixmap {path_filename}")
                        return
                    # Preserve device pixel ratio
                    base_pixmap_copy.setDevicePixelRatio(pixmap.devicePixelRatio())
                    label.base_pixmap = base_pixmap_copy
                except Exception as e:
                    logging.error(f"[THUMBNAIL] Failed to create deep copy of base_pixmap for {path_filename}: {e}")
                    return
                # Store actual thumbnail dimensions on label for bbox painting
                label._thumb_width = thumb_width  # type: ignore[attr-defined]
                label._thumb_height = thumb_height  # type: ignore[attr-defined]
                label._thumb_offset_x = offset_x  # type: ignore[attr-defined]
                label._thumb_offset_y = offset_y  # type: ignore[attr-defined]
                # Final verification: ensure label filename still matches (prevent race conditions)
                final_filename_check = getattr(label, "_thumb_filename", None)
                if final_filename_check != path_filename:
                    logging.error(f"[THUMBNAIL] Race condition: Label filename changed from {path_filename} to {final_filename_check} during processing. Skipping.")
                    return

                # Create a DEEP copy to draw overlays on - ensure each label gets its own pixmap
                # Convert to QImage and back to QPixmap to force a true deep copy (prevents sharing)
                try:
                    # Convert pixmap to image (forces deep copy of pixel data)
                    img = pixmap.toImage()
                    if img.isNull():
                        logging.error(f"[THUMBNAIL] Failed to convert pixmap to image for {path_filename}")
                        return
                    # Create new pixmap from image (truly independent copy)
                    overlay = QPixmap.fromImage(img)
                    if overlay.isNull():
                        logging.error(f"[THUMBNAIL] Failed to create pixmap from image for {path_filename}")
                        return
                    # Preserve device pixel ratio
                    overlay.setDevicePixelRatio(pixmap.devicePixelRatio())
                except Exception as e:
                    logging.error(f"[THUMBNAIL] Failed to create deep copy of pixmap for {path_filename}: {e}")
                    return

                # Apply overlays for current state (bboxes, badges)
                try:
                    # Find current grid position
                    current_pos = None
                    for (r, c), lbl in self.label_refs.items():
                        if lbl == label:
                            current_pos = (r, c)
                            break

                    # Verify no other label at this position has a different filename
                    if current_pos:
                        other_label = self.label_refs.get(current_pos)
                        if other_label and other_label != label:
                            logging.error(f"[THUMBNAIL] Position {current_pos} has different label! Expected {id(label)}, found {id(other_label)}")
                            return

                    # Log pixmap IDs to track if same pixmap is being used for multiple labels
                    logging.debug(f"[THUMBNAIL] Setting pixmap for {path_filename} at grid_pos={current_pos}, label_id={id(label)}, label_filename={getattr(label, '_thumb_filename', None)}, pixmap_id={id(pixmap)}, overlay_id={id(overlay)}, label visible={label.isVisible()}, pixmap size={pixmap.width()}x{pixmap.height()}")
                    self._set_label_pixmap(label, overlay, ctx="thumbnail")
                except Exception:
                    logging.exception(f"[THUMBNAIL] Failed to set label pixmap with overlays for {path_filename}")
            else:
                label.clear()
        except ValueError:
            # Fail fast on ValueError - these indicate bugs that should crash the app
            raise
        except Exception:
            logging.exception("Error in _on_thumbnail_loaded")

    def _on_detection_done(self, filename, detections):
        """Detection complete; state cached in browser_state_changed handler."""
        pass

    def _on_detection_ready(self, filename):
        """Detection ready; trigger state refresh to update thumbnails with new bounding boxes."""
        # Trigger state snapshot update so detections are included in browser_state_changed
        # This will cause _on_browser_state_changed to refresh thumbnails with new bboxes
        from PySide6.QtCore import QTimer
        logging.debug(f"[BBOX-WIDGET] Detection ready for {filename}, triggering state refresh")
        QTimer.singleShot(0, self.viewmodel._emit_state_snapshot)
        # Also trigger immediate badge refresh to update overlays
        QTimer.singleShot(100, self._refresh_thumbnail_badges)

    def _on_label_changed(self, filename: str, label: str):
        """Handler for `viewmodel.label_changed` — display a toast for auto-assigned labels.

        The repository stores label source; query it to decide whether to show an auto toast.
        """
        try:
            repo = getattr(self.viewmodel.model, "_repo", None)
            if not repo:
                return
            source = repo.get_label_source(filename)
            if source == "auto":
                prob = self._get_current_prediction_prob(filename)
                if prob is None:
                    txt = f"Auto-labelled {label.upper()}"
                else:
                    txt = f"Auto-labelled {label.upper()} (p={prob:.2f})"
                self._show_toast(txt)
        except Exception:
            logging.exception("Error in _on_label_changed")

    def _schedule_update_label_icon(
        self, thumb_label, display_label, fname, is_auto, prob, detections, retries=3, delay_ms=150
    ):
        """Overlay system removed; detection data cached in state."""
        pass

    def _on_progress_changed(self, current, total):
        # PERFORMANCE: Use same debouncing as _on_task_progress
        if not hasattr(self, "_progress_update_timer"):
            from PySide6.QtCore import QTimer
            self._progress_update_timer = QTimer()
            self._progress_update_timer.setSingleShot(True)
            self._progress_update_pending = {}
            self._progress_update_timer.timeout.connect(self._apply_progress_update)

        # Store pending update
        self._progress_update_pending = {
            "name": "Loading images",
            "current": current,
            "total": total,
            "detail": None
        }

        # Restart timer (debounce to max 10 updates/sec)
        if not self._progress_update_timer.isActive():
            self._progress_update_timer.start(100)

        # Hide when complete (not debounced - immediate)
        if current >= total:
            self.progress_bar.hide()

    def _on_selected_image_changed(self, path):
        """Legacy signal: just refresh highlights (selection model already updated)."""
        self._update_all_highlights()

    def _on_has_selected_image_changed(self, has):
        self.open_btn.setEnabled(has)

    def _on_open_in_viewer(self):
        self.viewmodel.open_selected_in_viewer()


    def _on_task_started(self, name):
        # Track active tasks
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.add(name)
        self.progress_bar.setFormat(f"{name}: starting...")
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # indeterminate until first progress
        self.progress_bar.show()
        if name == "training":
            self.cancel_train_btn.setEnabled(True)

    def _on_task_progress(self, name, current, total, detail=None):
        # PERFORMANCE: Debounce progress bar updates to avoid excessive setValue calls (1,903 calls -> reduced)
        # This method is called from main thread via Qt signals (thread-safe)
        if not hasattr(self, "_progress_update_timer"):
            from PySide6.QtCore import QTimer
            self._progress_update_timer = QTimer(self)  # Parent to self to ensure main thread
            self._progress_update_timer.setSingleShot(True)
            self._progress_update_pending = {}
            self._progress_update_timer.timeout.connect(self._apply_progress_update)

        # Store pending update
        self._progress_update_pending = {
            "name": name,
            "current": current,
            "total": total,
            "detail": detail
        }

        # Restart timer (debounce to max 10 updates/sec)
        # Timer operations are safe here since this is called from main thread via signals
        try:
            if not self._progress_update_timer.isActive():
                self._progress_update_timer.start(100)
        except RuntimeError:
            # Timer was deleted (app closing), ignore
            pass

    def _apply_progress_update(self):
        """Apply debounced progress bar update."""
        if not self._progress_update_pending:
            return

        pending = self._progress_update_pending
        name = pending["name"]
        current = pending["current"]
        total = pending["total"]
        detail = pending.get("detail")

        if total > 0:
            if self.progress_bar.maximum() != total:
                self.progress_bar.setMaximum(total)
                self.progress_bar.setMinimum(0)
            self.progress_bar.setValue(current)
            txt = f"{name}: {detail or ''} {current}/{total}".strip()
            self.progress_bar.setFormat(txt)
        else:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setMinimum(0)
            txt = f"{name}: {detail or 'working...'}".strip()
            self.progress_bar.setFormat(txt)
        # Ensure progress bar is visible during progress updates
        if not self.progress_bar.isVisible():
            self.progress_bar.show()

        self._progress_update_pending = {}

    def _on_task_finished(self, name, ok):
        # Track active tasks
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.discard(name)
        self.progress_bar.setFormat(f"{name}: {'done' if ok else 'failed'}")
        if name == "training":
            # Re-enable train button
            self.train_btn.setEnabled(True)
            self.cancel_train_btn.setEnabled(False)
        # Only hide progress bar if no tasks are active
        from PySide6.QtCore import QTimer
        if not self._active_tasks:
            # Hide progress bar after a short delay to show completion message
            QTimer.singleShot(800, self.progress_bar.hide)
        else:
            # Update format to show next active task
            if self._active_tasks:
                next_task = next(iter(self._active_tasks))
                self.progress_bar.setFormat(f"{next_task}: running...")

    def _on_model_stats_changed(self, stats):
        """Update the model performance display with training metrics.

        Populates the headline metric labels and the compact top-features list.
        """
        try:
            # Headline metrics (defensive key lookup)
            if not stats:
                self.metric_auc.setText("AUC: —")
                self.metric_precision.setText("Precision: —")
                self.metric_f1.setText("F1: —")
                self._clear_layout(self.top_features_list_layout)
                return

            auc = stats.get("auc") or stats.get("roc_auc") or stats.get("cv_accuracy_mean")
            precision = stats.get("precision")
            f1 = stats.get("f1") or stats.get("f1_score")

            if auc is None:
                self.metric_auc.setText("AUC: —")
            else:
                try:
                    self.metric_auc.setText(f"AUC: {float(auc) * 100:.2f}%")
                except Exception:
                    self.metric_auc.setText(f"AUC: {auc}")

            if precision is None:
                self.metric_precision.setText("Precision: —")
            else:
                try:
                    self.metric_precision.setText(f"Precision: {float(precision) * 100:.2f}%")
                except Exception:
                    self.metric_precision.setText(f"Precision: {precision}")

            if f1 is None:
                self.metric_f1.setText("F1: —")
            else:
                try:
                    self.metric_f1.setText(f"F1: {float(f1) * 100:.2f}%")
                except Exception:
                    self.metric_f1.setText(f"F1: {f1}")

            # Training metrics (loss, iterations, patience)
            final_loss = stats.get("final_loss")
            if final_loss is None:
                self.metric_loss.setText("Loss: —")
            else:
                try:
                    # Format loss with fewer decimals for readability
                    self.metric_loss.setText(f"Loss: {float(final_loss):.4f}")
                except Exception:
                    self.metric_loss.setText(f"Loss: {final_loss}")

            iterations = stats.get("iterations")
            if iterations is None:
                self.metric_iterations.setText("Iter: —")
            else:
                try:
                    # Shortened label to save space
                    self.metric_iterations.setText(f"Iter: {int(iterations)}")
                except Exception:
                    self.metric_iterations.setText(f"Iter: {iterations}")

            patience = stats.get("patience")
            if patience is None:
                self.metric_patience.setText("Pat: —")
            else:
                try:
                    # Shortened label to save space
                    self.metric_patience.setText(f"Pat: {int(patience)}")
                except Exception:
                    self.metric_patience.setText(f"Pat: {patience}")

            # Top features
            self._clear_layout(self.top_features_list_layout)
            fi = stats.get("feature_importances") or []
            # Get metadata from stats for interaction/embedding feature names
            meta = stats.get("model_metadata", {})
            feature_names = _get_feature_names(meta=meta)
            if fi:
                # Normalize importances to percentages (sum to 100%)
                total_importance = sum(imp for _, imp in fi) if fi else 1.0
                # Create compact chips for top features for better readability
                # Show top 20 to ensure COCO classes are visible (COCO: Person is 18th)
                for idx, (feat_idx, importance) in enumerate(fi[:20], 1):
                    name = feature_names.get(feat_idx, f"Feature {feat_idx}")
                    percentage = (importance / total_importance * 100) if total_importance > 0 else 0.0
                    chip = QLabel(f"{idx}. {name} — {percentage:.1f}%")
                    chip.setToolTip(name)
                    chip.setStyleSheet("""
                        QLabel {
                            background: #e3f2fd;
                            color: #1976d2;
                            padding: 6px 12px;
                            border-radius: 16px;
                            font-size: 11px;
                            font-weight: 500;
                            border: 1px solid #bbdefb;
                        }
                    """)
                    chip.setFixedHeight(26)
                    chip.setContentsMargins(2, 0, 2, 0)
                    self.top_features_list_layout.addWidget(chip)
        except Exception:
            logging.exception("Failed to update model stats display")
        # Training chart removed

    def _update_directory_display(self):
        """Update the directory label display with current directory path."""
        directory = self.viewmodel.model.directory if self.viewmodel else ""
        if directory:
            # Show shortened path (last 2 components)
            from pathlib import Path

            path_obj = Path(directory)
            if len(path_obj.parts) > 2:
                display = f".../{path_obj.parts[-2]}/{path_obj.parts[-1]}"
            else:
                display = directory
            self.dir_display.setText(display)
            self.dir_display.setToolTip(directory)
        else:
            self.dir_display.setText("No directory")

    def _on_switch_directory(self):
        """Open directory selection dialog and switch directories."""
        from PySide6.QtWidgets import QFileDialog

        current_dir = self.viewmodel.model.directory if self.viewmodel else str(Path.home() / "Pictures")

        new_dir = QFileDialog.getExistingDirectory(
            self, "Select Photo Directory", current_dir, QFileDialog.Option.ShowDirsOnly
        )

        if new_dir:
            success = change_directory(self.viewmodel, new_dir)
            if success:
                self._update_directory_display()
                # Clear selections and metadata
                self.selected_filenames.clear()
                self.exif_view.clear()
                self.open_btn.setEnabled(False)
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.critical(self, "Error", f"Failed to switch to directory:\n{new_dir}")

    def _on_directory_changed(self, new_dir):
        """Handle directory change signal from viewmodel."""
        self._update_directory_display()

    def _force_reapply_overlays(self):
        """Overlay system removed; detection data cached in state."""
        pass

    def _on_browser_state_changed(self, state):
        """Handle browser state changes, particularly prediction updates."""
        if not state:
            return

        self._last_browser_state = state

        # Debug: log group_info if present
        group_info = getattr(state, "group_info", {})
        detected_objects = getattr(state, "detected_objects", {})
        if group_info:
            group_count = len(set(g.get("group_id", 0) for g in group_info.values() if g.get("group_id") is not None))
            best_count = sum(1 for g in group_info.values() if g.get("is_group_best", False))
            multi_photo_groups = sum(1 for g in group_info.values() if g.get("group_size", 1) > 1)
            logging.info(f"[view] Browser state updated: {len(group_info)} photos with group_info, {group_count} groups, {best_count} best picks, {multi_photo_groups} photos in multi-photo groups")
        if detected_objects:
            logging.info(f"[view] Browser state updated: {len(detected_objects)} photos with detected_objects")
        # Always trigger badge refresh when state changes (to show groups, bboxes, badges)
        self._refresh_thumbnail_badges()
        pred_probs = getattr(state, "predicted_probabilities", {})
        pred_labels = getattr(state, "predicted_labels", {})
        detected_objects = getattr(state, "detected_objects", {})
        filtered_images = getattr(state, "filtered_images", None)

        # Force relayout when predictions are complete to ensure final sorted order is applied
        # This handles the case where initial relayout happened before predictions,
        # so we need to re-layout with the final uncertainty-based sort
        if filtered_images and len(pred_probs) > 0:
            total_images = len(filtered_images)
            predictions_complete = len(pred_probs) >= total_images * 0.9  # 90% have predictions

            # Track if we've already forced a relayout for this prediction completion
            if not hasattr(self, "_predictions_complete_relayout_forced"):
                self._predictions_complete_relayout_forced = False

            if predictions_complete and not self._predictions_complete_relayout_forced:
                # Reset last order to force relayout with final sorted order
                self._last_filtered_images_order = None
                self._predictions_complete_relayout_forced = True  # Only force once

        # Auto-select image with most bounding boxes for testing
        if detected_objects and not hasattr(self, "_auto_selected_most_boxes"):
            max_count = 0
            max_file = None
            for filename, detections in detected_objects.items():
                count = len(detections) if detections else 0
                if count > max_count:
                    max_count = count
                    max_file = filename
            if max_file and max_count > 0:
                logging.debug(f"[DEBUG] Auto-selecting image with most detections: {max_file} ({max_count} boxes)")
                self.viewmodel.select_image(max_file)
                self._auto_selected_most_boxes = True  # Only do this once

        # Update grid visibility based on filtered images
        if filtered_images is not None:
            filtered_set = set(filtered_images)
            logging.debug(f"[view] Updating grid visibility: {len(filtered_images)} filtered images out of {len(self.label_refs)} total")
            visible_count = 0
            for (row, col), thumb_label in list(self.label_refs.items()):
                f_path = thumb_label.toolTip()
                f_name = os.path.basename(f_path) if f_path else None
                if f_name:
                    should_show = f_name in filtered_set
                    thumb_label.setVisible(should_show)
                    if should_show:
                        visible_count += 1
            logging.debug(f"[view] Made {visible_count} thumbnails visible, {len(self.label_refs) - visible_count} hidden")
            # Only force relayout if visibility actually changed (not on every state update)
            # Debounce relayout to avoid excessive calls
            if not hasattr(self, "_visibility_relayout_timer"):
                from PySide6.QtCore import QTimer
                self._visibility_relayout_timer = QTimer()
                self._visibility_relayout_timer.setSingleShot(True)
                self._visibility_relayout_timer.timeout.connect(lambda: self._relayout_grid())
            # Restart timer on each visibility change (debounce) - only relayout if timer fires
            self._visibility_relayout_timer.stop()
            self._visibility_relayout_timer.start(200)  # 200ms debounce (increased from 100ms)

        # Refresh label badges on all thumbnails since labels may have changed
        # This ensures all thumbnails show updated predictions after retraining
        # Debounce badge refresh to avoid excessive repaints (high CPU/memory)
        logging.debug(f"[view] State changed: {len(pred_probs)} predicted probabilities")
        # Use debounce timer to coalesce rapid state updates
        if not hasattr(self, "_badge_refresh_timer"):
            from PySide6.QtCore import QTimer
            self._badge_refresh_timer = QTimer()
            self._badge_refresh_timer.setSingleShot(True)
            self._badge_refresh_timer.timeout.connect(self._refresh_thumbnail_badges)
        if not self._badge_refresh_timer.isActive():
            self._badge_refresh_timer.start(200)  # 200ms debounce

        # If any detection entries include bounding-box coordinates, force a
        # full refresh so all thumbnails get their overlays painted. Previously
        # we only refreshed all thumbnails the first time any detections were
        # present which meant later-arriving bbox data (from background tasks)
        # would not be applied to thumbnails because we set `_bboxes_drawn` and
        # only updated the primary selection thereafter. Detect bbox-bearing
        # entries and redraw all thumbnails when they appear.
        def _has_any_bbox(dmap):
            return any(
                "bbox" in d and d.get("bbox") for dets in (dmap or {}).values() for d in dets if isinstance(d, dict)
            )

        has_bbox = _has_any_bbox(detected_objects)
        if has_bbox:
            # draw bounding boxes on all thumbnails so overlays appear
            # Trigger badge refresh which will update widget overlays
            self._bboxes_drawn = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_thumbnail_badges)
        else:
            # preserve legacy behaviour for the initial non-bbox summary case
            update_all = not hasattr(self, "_bboxes_drawn") and any(detected_objects.values())
            if update_all:
                self._bboxes_drawn = True
                for (row, col), thumb_label in self.label_refs.items():
                    f_path = thumb_label.toolTip()
                    f_name = os.path.basename(f_path)
                    pred_prob = pred_probs.get(f_name, 0)
                    current_label = self.viewmodel.model.get_state(f_path) if f_path else None
                    display_label = current_label or pred_labels.get(f_name, "")
                    is_auto = self._is_auto_labeled(f_name)
                    objects = detected_objects.get(f_name, [])
            else:
                # Only update the thumbnail for the primary selected image
                primary = self.viewmodel.selection_model.primary()
                if primary:
                    primary_name = os.path.basename(primary)
                    for (row, col), thumb_label in self.label_refs.items():
                        f_path = thumb_label.toolTip()
                        f_name = os.path.basename(f_path)
                        if f_name == primary_name:
                            pred_prob = pred_probs.get(f_name, 0)
                            current_label = self.viewmodel.model.get_state(f_path) if f_path else None
                            display_label = current_label or pred_labels.get(f_name, "")
                            # Overlay system removed; detection data cached in state
                            break

        # Refresh EXIF for current primary selection
        exif_data = getattr(self.viewmodel, "exif", {})
        self._on_exif_changed(exif_data)

        # Refresh object detection display
        primary = self.viewmodel.selection_model.primary()
        if primary:
            self._update_object_detection_display(primary)

    def _update_prediction_display(self, filename: str, probability: float, predicted_label: str):
        """Update a single thumbnail with new prediction (called incrementally during batch prediction)."""
        # Find the label widget for this filename
        for (row, col), thumb_label in self.label_refs.items():
            f_path = thumb_label.toolTip()
            f_name = os.path.basename(f_path)

            if f_name == filename:
                # Only display prediction if no manual label
                current_label = self.viewmodel.model.get_state(f_path) if f_path else None
                if not current_label:
                    # Display the predicted label
                    # Overlay system removed; detection data cached in state
                    pass
                break

    def _get_current_prediction_prob(self, filename: str):
        """Return freshest prediction probability (keep prob 0..1) for a filename.
        Checks (1) last browser state snapshot, (2) auto label manager, (3) legacy cache.
        Returns None if unavailable or NaN."""
        if not filename:
            return None
        fname = os.path.basename(filename)
        # 1. snapshot
        if self._state:
            try:
                p = getattr(self._state, "predicted_probabilities", {}).get(fname)
                if p is not None and p == p:  # not None and not NaN
                    return p
            except Exception:
                logging.exception("Error updating view thumbnail")
                raise
        # 2. auto manager
        auto = getattr(self.viewmodel, "_auto", None)
        if auto:
            try:
                p = getattr(auto, "predicted_probabilities", {}).get(fname)
                if p is not None and p == p:  # not None and not NaN
                    return p
            except Exception:
                logging.exception("Error loading view metadata")
                raise
        # 3. legacy viewmodel cache
        probs = getattr(self.viewmodel, "_predicted_probabilities", {})
        p = probs.get(fname)
        if p is not None and p == p:  # not None and not NaN
            return p
        return None

    def _navigate_selection(self, delta: int):
        """Move primary selection by delta in the current filtered image list."""
        try:
            imgs = self.viewmodel.current_filtered_images() or list(self.viewmodel.images)
            if not imgs:
                return

            # Determine current index (basename-aware)
            primary = self.viewmodel.selection_model.primary()
            current_name = os.path.basename(primary) if primary else None

            if current_name and current_name in imgs:
                idx = imgs.index(current_name)
            else:
                # No current selection: start at 0 or end depending on delta
                idx = 0 if delta >= 0 else len(imgs) - 1

            new_idx = max(0, min(len(imgs) - 1, idx + delta))
            new_fname = imgs[new_idx]
            # Use viewmodel to select by filename
            self.viewmodel.select_image(new_fname)
        except Exception:
            # Be defensive in UI navigation
            logging.exception("_navigate_selection failed")

    def _show_shortcuts_help(self):
        """Display an enhanced help dialog listing available keyboard shortcuts."""
        try:
            from PySide6.QtWidgets import QMessageBox

            txt = (
                "📋 Keyboard Shortcuts\n\n"
                "🏷️  Labeling:\n"
                "  K — Keep (label)\n"
                "  T — Trash (label)\n"
                "🎯 Navigation:\n"
                "  Arrow keys — Move selection\n"
                "  Ctrl+Click — Multi-select\n"
                "  Shift+Click — Range select\n\n"
                "🎨 Visual Indicators:\n"
                "  🟢 Green border — Keep\n"
                "  🔴 Red border — Trash\n"
                "  ⚪ Gray border — Unlabeled\n"
                "  🔵 Blue border — Selected\n\n"
                "  ? — Show this help\n\n"
                "💡 Tip: Use keyboard shortcuts for faster photo triage!"
            )
            QMessageBox.information(self, "Keyboard Shortcuts", txt)
        except Exception:
            logging.exception("Failed to show shortcuts help")

    def resizeEvent(self, event):
        """Handle window resize to relayout grid."""
        super().resizeEvent(event)
        # Debounce relayout to avoid excessive calls during resize
        if not hasattr(self, "_window_resize_relayout_timer"):
            from PySide6.QtCore import QTimer
            self._window_resize_relayout_timer = QTimer()
            self._window_resize_relayout_timer.setSingleShot(True)
            self._window_resize_relayout_timer.timeout.connect(self._relayout_grid)
        # Restart timer on each resize event (debounce) - stop and restart to reset timeout
        self._window_resize_relayout_timer.stop()
        self._window_resize_relayout_timer.start(150)  # 150ms debounce

    def eventFilter(self, obj, event):
        # PERFORMANCE: Early return for unhandled events to avoid expensive checks
        # Cache event type to avoid repeated calls (158k calls -> significant savings)
        event_type = event.type()
        
        # OPTIMIZATION: Fast path - return False immediately for unhandled event types
        # Most events are not handled by this filter, so early return saves significant time
        if event_type not in (QEvent.Type.Resize, QEvent.Type.Enter, QEvent.Type.Leave, QEvent.Type.ContextMenu):
            return False

        # Handle resize events on scroll area viewport
        if obj == self.scroll_area.viewport() and event_type == QEvent.Type.Resize:
            # Debounce relayout to avoid excessive calls during resize
            if not hasattr(self, "_resize_relayout_timer"):
                from PySide6.QtCore import QTimer
                self._resize_relayout_timer = QTimer()
                self._resize_relayout_timer.setSingleShot(True)
                self._resize_relayout_timer.timeout.connect(self._relayout_grid)
            # Restart timer on each resize event (debounce) - stop and restart to reset timeout
            self._resize_relayout_timer.stop()
            self._resize_relayout_timer.start(150)  # 150ms debounce
            return False

        # PERFORMANCE: Only check for thumbnail events if object has _thumb_filename attribute
        # This avoids hasattr() calls for most events (majority of 158k calls)
        if not hasattr(obj, "_thumb_filename"):
            return super().eventFilter(obj, event)

        # Handle context menu from thumbnail QLabel
        try:
            # Hover previews disabled to prevent window accumulation
            # Context menu from thumbnail QLabel
            if event_type == QEvent.Type.Enter:
                # Disabled hover preview to prevent multiple windows
                pass
            elif event_type == QEvent.Type.Leave:
                try:
                    # Close any existing hover tooltip
                    self._close_hover_tooltip()
                except Exception:
                    logging.exception("Failed to close transient preview in eventFilter")
            elif event_type == QEvent.Type.ContextMenu:
                try:
                    f = getattr(obj, "_thumb_filename", None)
                    if f:
                        self._show_thumb_context_menu(obj, f, event.pos())
                        return True
                except Exception:
                    logging.exception("Failed to show context menu in eventFilter")
        except Exception:
            logging.exception("Failed in eventFilter")
        return super().eventFilter(obj, event)

    def _show_thumb_context_menu(self, label, filename, pos):
        try:
            menu = QMenu(self)

            def _open_this():
                try:
                    # select the image first so the viewmodel has a primary selection
                    self.viewmodel.select_image(filename)
                except Exception:
                    logging.exception("Failed to select image in context menu")
                try:
                    self.viewmodel.open_selected_in_viewer()
                except Exception:
                    logging.exception("Failed to open selected image in viewer")

            menu.addAction("Open", _open_this)
            menu.addAction("Open in Viewer", _open_this)
            menu.addSeparator()
            menu.addAction("Keep", lambda: self.viewmodel.set_label("keep", filename))
            menu.addAction("Trash", lambda: self.viewmodel.set_label("trash", filename))
            menu.addSeparator()
            menu.addAction("Show EXIF", lambda: self._select_and_show_exif(filename))
            menu.addAction(
                "Copy Path", lambda: QGuiApplication.clipboard().setText(self.viewmodel.model.get_image_path(filename))
            )
            # Translate pos from label-local to global
            global_pos = label.mapToGlobal(pos)
            menu.exec(global_pos)
        except Exception:
            logging.exception("Failed to show context menu")

    def _select_and_show_exif(self, filename):
        try:
            # select image in viewmodel and force exif refresh
            self.viewmodel.select_image(filename)
            path = self.viewmodel.model.get_image_path(filename)
            self.lazy_loader.get_exif_lazy(path, lambda p, e: self._on_exif_changed(e))
        except Exception:
            logging.exception("Failed to select/show EXIF")

    def _close_hover_tooltip(self):
        """Close any existing hover tooltip."""
        if self._hover_tooltip is not None:
            try:
                self._hover_tooltip.close()
                self._hover_tooltip.deleteLater()
            except Exception:
                pass
            self._hover_tooltip = None

    def _show_hover_preview(self, label):
        # Disabled to prevent window accumulation
        # Hover previews are no longer shown
        pass
