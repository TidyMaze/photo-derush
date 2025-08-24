import sys
import logging
from PySide6.QtWidgets import QApplication
import qdarktheme
import os
import json
from src.viewmodel import PhotoViewModel
from src.view import PhotoView

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')
MAX_IMAGES = 100

def load_last_dir():
    try:
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
            last_dir = data.get('last_dir')
            if last_dir and os.path.isdir(last_dir):
                return last_dir
    except Exception:
        pass
    return os.path.expanduser('~')  # fallback

def save_last_dir(path):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'last_dir': path}, f)
    except Exception as e:
        logging.warning(f"Could not save config: {e}")

def main():
    logging.info("App main() starting...")
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    last_dir = load_last_dir()  # Use config, not forced cwd
    logging.info(f"Loaded last_dir from config: {last_dir}")
    viewmodel = PhotoViewModel(last_dir, max_images=MAX_IMAGES)
    logging.info("PhotoViewModel created, about to create PhotoView and call load_images()")
    view = PhotoView(viewmodel)
    view.show()
    viewmodel.load_images()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
