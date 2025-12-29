#!/usr/bin/env python3
import os
import sys
import logging
os.environ.setdefault('QT_LOGGING_RULES', 'qt.*=false')
os.environ.setdefault('DETECTION_WORKER', '0')
# Try offscreen platform to avoid opening windows when possible
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QPixmap, QPainter, QColor
from src.view_helpers import update_label_icon

# configure logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.DEBUG), format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

def make_pixmap(w, h, color=(200,200,200)):
    p = QPixmap(w, h)
    p.fill(QColor(*color))
    painter = QPainter(p)
    painter.setPen(QColor(0,0,0))
    painter.drawRect(0,0,w-1,h-1)
    painter.end()
    return p


def main():
    app = QApplication(sys.argv)
    label = QLabel()
    size = 128
    label.setFixedSize(size, size)
    # create an original pixmap (square)
    pix = make_pixmap(size, size)
    label.original_pixmap = pix
    label.base_pixmap = pix
    # simulate a centered image of 100x80 inside 128x128 square
    img_w, img_h = 100, 80
    x = (size - img_w) // 2
    y = (size - img_h) // 2
    label._overlay_image_offset = (x, y, img_w, img_h)

    # Create a detection that was computed on an 800x600 image
    det = {
        'class': 'person',
        'confidence': 0.95,
        'bbox': [100.0, 50.0, 300.0, 400.0],
        'det_w': 800,
        'det_h': 600,
    }

    logging.info('Running update_label_icon debug call')
    update_label_icon(label, 'keep', filename='test.jpg', is_auto=False, prediction_prob=None, objects=[det])

    out_dir = os.path.join('.cache')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'debug_overlay.png')
    try:
        label.pixmap().save(out_path)
        logging.info('Saved debug overlay image to %s', out_path)
    except Exception:
        logging.exception('Failed saving debug overlay image')

    # Print mapped bbox from function directly for quick check
    from src.view_helpers import map_bbox_to_thumbnail
    rx = map_bbox_to_thumbnail(det['bbox'], det['det_w'], det['det_h'], img_w, img_h, x, y)
    logging.info('map_bbox_to_thumbnail returned: %s', rx)

    # Exit
    sys.exit(0)

if __name__ == '__main__':
    main()
