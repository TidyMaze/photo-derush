"""
PySide6 port of the Lightroom UI from main.py (Tkinter version).
- Main window: QMainWindow with left (image grid) and right (info) panels
- Image grid: QScrollArea + QGridLayout, thumbnails, selection, metrics overlay
- Full image viewer: QDialog or QMainWindow, closes on click/ESC
- All event handling and image display is Qt idiomatic
"""
import sys
from PySide6.QtWidgets import QApplication
from .main_window import LightroomMainWindow
from .viewer import open_full_image_qt
from PySide6.QtCore import QTimer
import threading
from precompute import prepare_images_and_groups, MAX_IMAGES as PREP_MAX, list_images
import logging
import queue

def show_lightroom_ui_qt(image_paths, directory, trashed_paths=None, trashed_dir=None, on_window_opened=None, image_info=None):
    app = QApplication.instance() or QApplication(sys.argv)
    # Apply global darkstyle
    import os
    qss_path = os.path.join(os.path.dirname(__file__), 'qdarkstyle.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            app.setStyleSheet(f.read())
    def get_sorted_images():
        return image_paths
    win = LightroomMainWindow(image_paths, directory, get_sorted_images, image_info=image_info)
    win.show()
    app.exec()

def show_lightroom_ui_qt_async(directory, max_images=PREP_MAX):
    app = QApplication.instance() or QApplication(sys.argv)
    import os
    qss_path = os.path.join(os.path.dirname(__file__), 'qdarkstyle.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            app.setStyleSheet(f.read())
    def empty_sorted():
        return []
    win = LightroomMainWindow([], directory, empty_sorted, image_info={})
    win.status.showMessage("Preparing images in background…")
    win.show()

    msg_queue = queue.Queue()
    stop_flag = {'done': False}

    def poll_queue():
        processed_any = False
        try:
            while True:
                msg = msg_queue.get_nowait()
                processed_any = True
                mtype = msg.get('type')
                if mtype == 'image':
                    img_name = msg['name']
                    logging.info("[AsyncLoad] (poll) Adding image %s", img_name)
                    if win.image_grid:
                        win.image_grid.add_image(img_name)
                    else:
                        logging.warning("[AsyncLoad] (poll) Image grid not ready, skipping %s", img_name)
                elif mtype == 'grouping':
                    logging.info("[AsyncLoad] (poll) Applying grouping metadata")
                    win.update_grouping(msg['image_info'])
                    win.status.showMessage(f"Loaded {len(win.image_grid.image_labels)} images (groups ready)")
                elif mtype == 'error':
                    win.status.showMessage(f"Background load failed: {msg['error']}")
                elif mtype == 'done':
                    stop_flag['done'] = True
                msg_queue.task_done()
        except queue.Empty:
            pass
        if not stop_flag['done']:
            QTimer.singleShot(50, poll_queue)
        elif processed_any:
            win.status.showMessage(win.status.currentMessage() or "Load complete")

    QTimer.singleShot(50, poll_queue)

    def worker():
        try:
            images = list_images(directory)
            subset = images[:max_images]
            logging.info("[AsyncLoad] (worker) Found %d images, streaming first %d", len(images), len(subset))
            # Provide sorted list early
            msg_queue.put({'type': 'sorted', 'list': subset})
            for img in subset:
                msg_queue.put({'type': 'image', 'name': img})
            # Hashing & grouping
            images2, image_info, stats = prepare_images_and_groups(directory, max_images)
            logging.info("[AsyncLoad] (worker) Grouping stats: %s", stats)
            msg_queue.put({'type': 'grouping', 'image_info': image_info})
            msg_queue.put({'type': 'done'})
        except Exception as e:
            logging.exception("[AsyncLoad] Worker failed: %s", e)
            msg_queue.put({'type': 'error', 'error': str(e)})
            msg_queue.put({'type': 'done'})

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.exec()
