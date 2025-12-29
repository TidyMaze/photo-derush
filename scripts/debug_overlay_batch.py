#!/usr/bin/env python3
import os, sys, logging
from PIL import Image

os.environ.setdefault('QT_LOGGING_RULES', 'qt.*=false')
os.environ.setdefault('DETECTION_WORKER', '0')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QPixmap, QPainter, QColor
from src.view_helpers import update_label_icon, map_bbox_to_thumbnail

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.DEBUG), format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

TEST_DIR = os.path.join(os.getcwd(), 'test_images')
if not os.path.isdir(TEST_DIR):
    logging.error('test_images directory not found: %s', TEST_DIR)
    sys.exit(1)

THUMB = 128

app = QApplication(sys.argv)

out_dir = os.path.join('.cache', 'debug_overlays')
os.makedirs(out_dir, exist_ok=True)

files = sorted([f for f in os.listdir(TEST_DIR) if os.path.isfile(os.path.join(TEST_DIR, f))])
if not files:
    logging.error('No images in test_images')
    sys.exit(1)

for fname in files:
    path = os.path.join(TEST_DIR, fname)
    try:
        with Image.open(path) as im:
            orig_w, orig_h = im.size
    except Exception:
        logging.exception('Failed to open image %s', path)
        continue

    # Create label and pixmap like view._on_thumbnail_loaded
    label = QLabel()
    label.setFixedSize(THUMB, THUMB)

    # Convert PIL image to QPixmap minimal (we'll just create a colored box)
    # Compute scaled size keeping aspect ratio
    ratio = min(THUMB / orig_w, THUMB / orig_h)
    scaled_w = max(1, int(orig_w * ratio))
    scaled_h = max(1, int(orig_h * ratio))

    from PySide6.QtGui import QPixmap as QPg, QPainter
    square = QPg(THUMB, THUMB)
    square.fill(QColor(50, 50, 60))
    p = QPainter(square)
    x = (THUMB - scaled_w) // 2
    y = (THUMB - scaled_h) // 2
    # draw placeholder inner rect representing scaled image
    p.fillRect(x, y, scaled_w, scaled_h, QColor(180, 180, 200))
    p.end()

    label.base_pixmap = square
    label.original_pixmap = square
    label._overlay_image_offset = (x, y, scaled_w, scaled_h)

    # Infer det_w/det_h like app does: if max(orig)>800, scale to 800
    max_size = 800
    if max(orig_w, orig_h) > max_size:
        r = max_size / float(max(orig_w, orig_h))
        det_w = int(orig_w * r)
        det_h = int(orig_h * r)
    else:
        det_w = orig_w
        det_h = orig_h

    # Make a sample bbox in detection coords: centered quarter box
    bx1 = det_w * 0.25
    by1 = det_h * 0.25
    bx2 = det_w * 0.75
    by2 = det_h * 0.75
    det = {'class': 'testobj', 'confidence': 0.99, 'bbox': [bx1, by1, bx2, by2], 'det_w': det_w, 'det_h': det_h}

    logging.info('Processing %s orig=%sx%s det=%sx%s thumb_image=%sx%s offset=(%s,%s)', fname, orig_w, orig_h, det_w, det_h, scaled_w, scaled_h, x, y)

    update_label_icon(label, 'keep', filename=fname, is_auto=False, prediction_prob=None, objects=[det])

    out_path = os.path.join(out_dir, fname + '.overlay.png')
    try:
        label.pixmap().save(out_path)
        logging.info('Saved overlay for %s -> %s', fname, out_path)
    except Exception:
        logging.exception('Failed to save overlay for %s', fname)

    # Also log mapped bbox coords
    mapped = map_bbox_to_thumbnail(det['bbox'], det['det_w'], det['det_h'], scaled_w, scaled_h, x, y)
    logging.info('%s mapped bbox -> %s', fname, mapped)

print('done')
sys.exit(0)
