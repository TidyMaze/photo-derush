#!/usr/bin/env python3
"""Run multiple concurrent detect requests using the library API to test serialization.

This script calls `get_objects_for_image` from multiple threads to emulate the GUI
threadpool workload while ensuring the new module-level lock prevents concurrent
native calls.
"""
import os
import sys
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src import object_detection as od


def worker(path, idx):
    try:
        objs = od.get_objects_for_image(path)
        print(f"worker {idx} ok: {len(objs)} objs")
        return True
    except Exception as e:
        print(f"worker {idx} error: {e}")
        return False


def main():
    os.environ.setdefault('DETECTION_DEVICE', 'cpu')
    images = sorted(glob.glob('test_images/*'))
    if not images:
        print('No test images found in test_images/; place some images and retry')
        return

    workers = 6
    loops = 4
    tasks = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = []
        for i in range(loops * workers):
            path = images[i % len(images)]
            futures.append(ex.submit(worker, path, i))
        ok = 0
        for f in as_completed(futures):
            if f.result():
                ok += 1

    print('done', ok, 'successful')
    print('elapsed', time.time()-start)


if __name__ == '__main__':
    main()
