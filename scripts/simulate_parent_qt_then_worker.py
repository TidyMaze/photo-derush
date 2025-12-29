#!/usr/bin/env python3
"""Simulate importing PySide6 in parent, then start DetectionWorker."""
import importlib
import time
import logging

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Import PySide6 top-level modules to simulate GUI app startup
        pyside = importlib.import_module('PySide6')
        logging.info('Imported PySide6 package: %s', getattr(pyside, '__file__', None))
        # Import QtWidgets to be closer to a real app
        qtwidgets = importlib.import_module('PySide6.QtWidgets')
        logging.info('Imported PySide6.QtWidgets: %s', getattr(qtwidgets, '__file__', None))
    except Exception as e:
        logging.exception('Failed to import PySide6 in parent: %s', e)

    # Now start the detection worker in the same process model as the app would
    try:
        from src.detection_worker import DetectionWorker
        w = DetectionWorker()
        w.start()
        time.sleep(3)
        w.stop()
        logging.info('Worker started/stopped successfully after parent imported PySide6')
    except Exception:
        logging.exception('Failed to run DetectionWorker')

if __name__ == '__main__':
    main()
