import json
import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

CONFIG_PATH = os.path.expanduser("~/.photo_app_config.json")

DEFAULT_SETTINGS = {
    "default_directory": os.path.expanduser("~"),
    "remember_last_dir": True,
    "max_images": 100,
    "thumbnail_size": 128,
    "images_per_row": 10,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"],
    "cache_enabled": True,
    "cache_size_mb": 500,
    "fullscreen_single_fraction": 0.92,
    "fullscreen_multi_fraction": 0.88,
    "fullscreen_side_margin": 32,
    "fullscreen_top_bottom_margin": 32,
    "fullscreen_theme": "dark",
    "auto_label_enabled": False,
    "auto_label_model_path": os.path.expanduser("~/.photo-derush-keep-trash-model.joblib"),
    "auto_label_keep_threshold": 0.8,
    "auto_label_trash_threshold": 0.2,
}

FILE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".raw", ".cr2", ".nef"]


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(450, 400)

        # Load current settings
        self.settings = self.load_settings()

        self.setup_ui()
        self.load_values()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_directory_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_file_types_group())
        layout.addWidget(self._build_performance_group())
        layout.addLayout(self._build_buttons())

    def _build_directory_group(self):
        """Build directory settings group."""
        group = QGroupBox("Directory Settings")
        layout = QFormLayout(group)

        self.default_dir_edit = QLineEdit()
        dir_browse_btn = QPushButton("Browse...")
        dir_browse_btn.clicked.connect(self.browse_default_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self.default_dir_edit, 1)
        dir_row.addWidget(dir_browse_btn)
        layout.addRow("Default Directory:", dir_row)

        self.remember_last_dir = QCheckBox("Remember last opened directory")
        layout.addRow(self.remember_last_dir)

        return group

    def _build_display_group(self):
        """Build display settings group."""
        group = QGroupBox("Display Settings")
        layout = QFormLayout(group)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(10, 1000)
        self.max_images_spin.setSuffix(" images")
        layout.addRow("Maximum Images to Load:", self.max_images_spin)

        self.thumb_size_spin = QSpinBox()
        self.thumb_size_spin.setRange(64, 512)
        self.thumb_size_spin.setSuffix(" pixels")
        layout.addRow("Thumbnail Size:", self.thumb_size_spin)

        self.images_per_row_spin = QSpinBox()
        self.images_per_row_spin.setRange(2, 20)
        layout.addRow("Images per Row:", self.images_per_row_spin)

        self.full_single_frac = QDoubleSpinBox()
        self.full_single_frac.setRange(0.50, 1.00)
        self.full_single_frac.setSingleStep(0.01)
        self.full_single_frac.setDecimals(2)
        layout.addRow("Fullscreen Single Fraction:", self.full_single_frac)

        self.full_multi_frac = QDoubleSpinBox()
        self.full_multi_frac.setRange(0.50, 1.00)
        self.full_multi_frac.setSingleStep(0.01)
        self.full_multi_frac.setDecimals(2)
        layout.addRow("Fullscreen Multi Fraction:", self.full_multi_frac)

        self.full_side_margin = QSpinBox()
        self.full_side_margin.setRange(0, 400)
        self.full_side_margin.setSuffix(" px")
        layout.addRow("Fullscreen Side Margin:", self.full_side_margin)

        self.full_tb_margin = QSpinBox()
        self.full_tb_margin.setRange(0, 400)
        self.full_tb_margin.setSuffix(" px")
        layout.addRow("Fullscreen Top/Bottom Margin:", self.full_tb_margin)

        self.full_theme_combo = QComboBox()
        self.full_theme_combo.addItems(["dark", "neutral", "checker"])
        layout.addRow("Fullscreen Theme:", self.full_theme_combo)

        return group

    def _build_file_types_group(self):
        """Build file types group."""
        group = QGroupBox("File Types")
        layout = QVBoxLayout(group)

        self.file_types = {}
        for ext in FILE_EXTENSIONS:
            checkbox = QCheckBox(ext.upper())
            self.file_types[ext] = checkbox
            layout.addWidget(checkbox)

        return group

    def _build_performance_group(self):
        """Build performance settings group."""
        group = QGroupBox("Performance")
        layout = QFormLayout(group)

        self.cache_enabled = QCheckBox("Enable thumbnail caching")
        layout.addRow(self.cache_enabled)

        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(100, 10000)
        self.cache_size_spin.setSuffix(" MB")
        layout.addRow("Cache Size Limit:", self.cache_size_spin)

        return group

    def _build_buttons(self):
        """Build button layout."""
        layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)

        layout.addWidget(reset_btn)
        layout.addStretch()
        layout.addWidget(cancel_btn)
        layout.addWidget(ok_btn)

        return layout

    def browse_default_dir(self):
        current_dir = self.default_dir_edit.text() or os.path.expanduser("~")
        directory = QFileDialog.getExistingDirectory(self, "Select Default Directory", current_dir)
        if directory:
            self.default_dir_edit.setText(directory)

    def load_settings(self):
        """Load settings from config file"""
        settings = DEFAULT_SETTINGS.copy()

        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH) as f:
                    saved_settings = json.load(f)
                    settings.update(saved_settings)
        except Exception as e:
            import logging

            logging.warning(f"Could not load settings: {e}")

        return settings

    def _collect_ui_values(self):
        """Collect current values from UI widgets."""
        return {
            "default_directory": self.default_dir_edit.text(),
            "remember_last_dir": self.remember_last_dir.isChecked(),
            "max_images": self.max_images_spin.value(),
            "thumbnail_size": self.thumb_size_spin.value(),
            "images_per_row": self.images_per_row_spin.value(),
            "allowed_extensions": [ext for ext, cb in self.file_types.items() if cb.isChecked()],
            "cache_enabled": self.cache_enabled.isChecked(),
            "cache_size_mb": self.cache_size_spin.value(),
            "fullscreen_single_fraction": round(self.full_single_frac.value(), 2),
            "fullscreen_multi_fraction": round(self.full_multi_frac.value(), 2),
            "fullscreen_side_margin": self.full_side_margin.value(),
            "fullscreen_top_bottom_margin": self.full_tb_margin.value(),
            "fullscreen_theme": self.full_theme_combo.currentText(),
        }

    def save_settings(self):
        """Save current settings to config file"""
        ui_settings = self._collect_ui_values()

        # Preserve auto-label settings (not exposed in UI)
        ui_settings.update(
            {
                "auto_label_enabled": self.settings.get("auto_label_enabled", False),
                "auto_label_model_path": self.settings.get(
                    "auto_label_model_path", DEFAULT_SETTINGS["auto_label_model_path"]
                ),
                "auto_label_keep_threshold": self.settings.get("auto_label_keep_threshold", 0.8),
                "auto_label_trash_threshold": self.settings.get("auto_label_trash_threshold", 0.2),
            }
        )

        try:
            existing = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH) as f:
                    existing = json.load(f)

            existing.update(ui_settings)

            with open(CONFIG_PATH, "w") as f:
                json.dump(existing, f, indent=2)

        except Exception as e:
            import logging

            logging.error(f"Could not save settings: {e}")

    def load_values(self):
        """Load current values into UI"""
        s = self.settings

        self.default_dir_edit.setText(s.get("default_directory", ""))
        self.remember_last_dir.setChecked(s.get("remember_last_dir", True))
        self.max_images_spin.setValue(s.get("max_images", 100))
        self.thumb_size_spin.setValue(s.get("thumbnail_size", 128))
        self.images_per_row_spin.setValue(s.get("images_per_row", 10))
        self.cache_enabled.setChecked(s.get("cache_enabled", True))
        self.cache_size_spin.setValue(s.get("cache_size_mb", 500))
        self.full_single_frac.setValue(s.get("fullscreen_single_fraction", 0.92))
        self.full_multi_frac.setValue(s.get("fullscreen_multi_fraction", 0.88))
        self.full_side_margin.setValue(s.get("fullscreen_side_margin", 32))
        self.full_tb_margin.setValue(s.get("fullscreen_top_bottom_margin", 32))

        # Load file type checkboxes
        allowed_exts = s.get("allowed_extensions", [".jpg", ".jpeg", ".png"])
        for ext, checkbox in self.file_types.items():
            checkbox.setChecked(ext in allowed_exts)

        # Load theme combo box
        theme = s.get("fullscreen_theme", "dark")
        if theme not in ["dark", "neutral", "checker"]:
            theme = "dark"
        idx = self.full_theme_combo.findText(theme)
        if idx >= 0:
            self.full_theme_combo.setCurrentIndex(idx)

    def reset_defaults(self):
        """Reset all settings to default values"""
        self.settings = DEFAULT_SETTINGS.copy()
        self.load_values()

    def accept(self):
        """Save settings when OK is clicked"""
        self.save_settings()
        super().accept()


def get_setting(key, default=None):
    """Utility function to get a single setting value"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                settings = json.load(f)
                return settings.get(key, default)
    except Exception:
        logging.exception("Failed to load settings")
        raise
    return default


def set_setting(key, value):
    """Utility function to set a single setting value"""
    try:
        settings = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                settings = json.load(f)

        settings[key] = value

        with open(CONFIG_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        import logging

        logging.error(f"Could not save setting {key}: {e}")
