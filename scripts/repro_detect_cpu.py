#!/usr/bin/env python3
"""Minimal repro: run object detection on a single test image forcing CPU device.
Run with: ./.venv/bin/python3 scripts/repro_detect_cpu.py
"""
import logging
from src import object_detection

logging.basicConfig(level=logging.INFO)

IMAGE = 'test_images/PXL_20250524_124137113.jpg'

if __name__ == '__main__':
    print('Using DETECTION_BACKEND=', object_detection.DETECTION_BACKEND)
    try:
        dets = object_detection.detect_objects(IMAGE, device='cpu', confidence_threshold=0.3)
        print('Detections:', dets)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
