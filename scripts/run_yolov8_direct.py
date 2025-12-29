#!/usr/bin/env python3
"""Direct runner that loads the detection backend and runs repeated detections.

Use this under lldb to capture native backtraces if YOLOv8 crashes.
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault('DETECTION_BACKEND', 'yolov8')
os.environ.setdefault('DETECTION_WORKER', '0')

from src.object_detection import _load_model, detect_objects

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    imgs = []
    d = os.path.join(os.getcwd(), 'test_images')
    for p in os.listdir(d):
        if p.lower().endswith(('.jpg', '.jpeg', '.png')):
            imgs.append(os.path.join(d, p))
    if not imgs:
        print('No test images found in', d); return

    img = imgs[0]
    logging.info('Loading model via _load_model')
    model, weights = _load_model('auto')
    logging.info('Model loaded, weights=%s', weights)

    # Run detection repeatedly to increase chances of repro
    for i in range(200):
        logging.info('Run %d: detecting %s', i+1, os.path.basename(img))
        try:
            res = detect_objects(img)
            logging.info('Run %d: got %d detections', i+1, len(res.get('detections', [])) if isinstance(res, dict) else 0)
        except Exception:
            logging.exception('Detection call failed')
        time.sleep(0.1)

if __name__ == '__main__':
    main()
