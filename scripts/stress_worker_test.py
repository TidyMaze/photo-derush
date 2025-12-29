#!/usr/bin/env python3
"""Stress test the detection worker by enqueueing many images rapidly.

Usage: run from project root with DETECTION_WORKER=1 DETECTION_BACKEND=yolov8
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.detection_worker import get_global_worker, stop_global_worker


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    img_dir = os.path.join(os.getcwd(), 'test_images')
    imgs = [os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        print('No images found in', img_dir); return

    w = get_global_worker()
    pending = {}
    start_time = time.time()
    # Enqueue images as fast as possible
    for i, img in enumerate(imgs * 3):
        req_id = w.enqueue(img, {})
        pending[req_id] = (img, time.time())
        if i % 10 == 0:
            logging.info(f'enqueued {i+1} requests, queue_length={w.queue_length()} pending={len(pending)}')
        # small bursty sleep to better simulate GUI adding many images quickly
        time.sleep(0.01)

    logging.info('All requests enqueued, waiting for results...')
    # collect with a timeout per-request
    try:
        while pending:
            for req_id in list(pending.keys()):
                try:
                    res = w.get_result(req_id, timeout=1.0)
                except TimeoutError:
                    continue
                img, t0 = pending.pop(req_id)
                logging.info(f'result for {req_id} image={os.path.basename(img)} got keys={list(res.keys())} elapsed={(time.time()-t0):.2f}s queue_length={w.queue_length()} pending={len(pending)}')
    finally:
        stop_global_worker()
    logging.info('Stress test finished in %.2fs' % (time.time() - start_time,))


if __name__ == '__main__':
    main()
