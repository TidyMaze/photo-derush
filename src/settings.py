import os
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QSpinBox, QCheckBox, QPushButton,
                               QFileDialog, QGroupBox, QComboBox, QFormLayout)
from PySide6.QtCore import Qt

CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')

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

        # Directory settings
        dir_group = QGroupBox("Directory Settings")
        dir_layout = QFormLayout(dir_group)

        self.default_dir_edit = QLineEdit()
        dir_browse_btn = QPushButton("Browse...")
        dir_browse_btn.clicked.connect(self.browse_default_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self.default_dir_edit, 1)
        dir_row.addWidget(dir_browse_btn)

        dir_layout.addRow("Default Directory:", dir_row)

        self.remember_last_dir = QCheckBox("Remember last opened directory")
        dir_layout.addRow(self.remember_last_dir)

        layout.addWidget(dir_group)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout(display_group)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(10, 1000)
        self.max_images_spin.setSuffix(" images")
        display_layout.addRow("Maximum Images to Load:", self.max_images_spin)

        self.thumb_size_spin = QSpinBox()
        self.thumb_size_spin.setRange(64, 512)
        self.thumb_size_spin.setSuffix(" pixels")
        display_layout.addRow("Thumbnail Size:", self.thumb_size_spin)

        self.images_per_row_spin = QSpinBox()
        self.images_per_row_spin.setRange(2, 20)
        display_layout.addRow("Images per Row:", self.images_per_row_spin)

        layout.addWidget(display_group)

        # File type settings
        file_group = QGroupBox("File Types")
        file_layout = QVBoxLayout(file_group)

        self.file_types = {}
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.raw', '.cr2', '.nef']:
            checkbox = QCheckBox(ext.upper())
            self.file_types[ext] = checkbox
            file_layout.addWidget(checkbox)

        layout.addWidget(file_group)

        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)

        self.cache_enabled = QCheckBox("Enable thumbnail caching")
        perf_layout.addRow(self.cache_enabled)

        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(100, 10000)
        self.cache_size_spin.setSuffix(" MB")
        perf_layout.addRow("Cache Size Limit:", self.cache_size_spin)

        layout.addWidget(perf_group)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def browse_default_dir(self):
        current_dir = self.default_dir_edit.text() or os.path.expanduser('~')
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Directory", current_dir)
        if directory:
            self.default_dir_edit.setText(directory)

    def load_settings(self):
        """Load settings from config file"""
        defaults = {
            'default_directory': os.path.expanduser('~'),
            'remember_last_dir': True,
            'max_images': 100,
            'thumbnail_size': 128,
            'images_per_row': 10,
            'allowed_extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            'cache_enabled': True,
            'cache_size_mb': 500
        }

        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    saved_settings = json.load(f)
                    defaults.update(saved_settings)
        except Exception as e:
            import logging
            logging.warning(f"Could not load settings: {e}")

        return defaults

    def save_settings(self):
        """Save current settings to config file"""
        settings = {
            'default_directory': self.default_dir_edit.text(),
            'remember_last_dir': self.remember_last_dir.isChecked(),
            'max_images': self.max_images_spin.value(),
            'thumbnail_size': self.thumb_size_spin.value(),
            'images_per_row': self.images_per_row_spin.value(),
            'allowed_extensions': [ext for ext, cb in self.file_types.items() if cb.isChecked()],
            'cache_enabled': self.cache_enabled.isChecked(),
            'cache_size_mb': self.cache_size_spin.value()
        }

        try:
            # Keep existing settings like last_dir if remember_last_dir is enabled
            existing = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    existing = json.load(f)

            # Merge with new settings
            existing.update(settings)

            with open(CONFIG_PATH, 'w') as f:
                json.dump(existing, f, indent=2)

        except Exception as e:
            import logging
            logging.error(f"Could not save settings: {e}")

    def load_values(self):
        """Load current values into UI"""
        self.default_dir_edit.setText(self.settings.get('default_directory', ''))
        self.remember_last_dir.setChecked(self.settings.get('remember_last_dir', True))
        self.max_images_spin.setValue(self.settings.get('max_images', 100))
        self.thumb_size_spin.setValue(self.settings.get('thumbnail_size', 128))
        self.images_per_row_spin.setValue(self.settings.get('images_per_row', 10))
        self.cache_enabled.setChecked(self.settings.get('cache_enabled', True))
        self.cache_size_spin.setValue(self.settings.get('cache_size_mb', 500))

        # Load file type checkboxes
        allowed_exts = self.settings.get('allowed_extensions', ['.jpg', '.jpeg', '.png'])
        for ext, checkbox in self.file_types.items():
            checkbox.setChecked(ext in allowed_exts)

    def reset_defaults(self):
        """Reset all settings to default values"""
        self.default_dir_edit.setText(os.path.expanduser('~'))
        self.remember_last_dir.setChecked(True)
        self.max_images_spin.setValue(100)
        self.thumb_size_spin.setValue(128)
        self.images_per_row_spin.setValue(10)
        self.cache_enabled.setChecked(True)
        self.cache_size_spin.setValue(500)

        # Reset file types to common image formats
        common_types = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        for ext, checkbox in self.file_types.items():
            checkbox.setChecked(ext in common_types)

    def accept(self):
        """Save settings when OK is clicked"""
        self.save_settings()
        super().accept()

def get_setting(key, default=None):
    """Utility function to get a single setting value"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                settings = json.load(f)
                return settings.get(key, default)
    except Exception:
        pass
    return default

def set_setting(key, value):
    """Utility function to set a single setting value"""
    try:
        settings = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                settings = json.load(f)

        settings[key] = value

        with open(CONFIG_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        import logging
        logging.error(f"Could not save setting {key}: {e}")
