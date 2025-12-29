#!/usr/bin/env python3
from src.detection_worker import DetectionWorker
import time

if __name__ == '__main__':
    w = DetectionWorker()
    w.start()
    # allow worker to initialize / write logs
    time.sleep(3)
    w.stop()
    print('worker run complete')
