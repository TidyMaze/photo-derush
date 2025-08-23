import sys
import logging
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow
import os
import json

logging.basicConfig(level=logging.INFO)
CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')

def load_last_dir():
    try:
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
            last_dir = data.get('last_dir')
            if last_dir and os.path.isdir(last_dir):
                return last_dir
    except Exception as e:
        logging.info(f"No previous config or invalid: {e}")
    return None

def save_last_dir(path):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'last_dir': path}, f)
    except Exception as e:
        logging.warning(f"Could not save config: {e}")

def get_image_files(directory):
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return [f for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in image_exts]

def main():
    app = QApplication(sys.argv)
    last_dir = load_last_dir()
    dir_path = QFileDialog.getExistingDirectory(None, "Select Directory", last_dir or "")
    if not dir_path:
        return
    save_last_dir(dir_path)
    image_files = get_image_files(dir_path)
    image_paths = [os.path.join(dir_path, f) for f in image_files]
    win = QMainWindow()
    win.setWindowTitle("Photo App - Placeholder Window")
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
