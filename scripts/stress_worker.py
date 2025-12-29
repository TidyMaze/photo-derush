"""Stress runner for the DetectionWorker.

This script starts a global detection worker and repeatedly enqueues detection
requests from `test_images/`. It's intentionally small and writes simple logs
so we can capture DYLD_PRINT_LIBRARIES output and let the attach-on-crash
monitor generate debugger backtraces if a native crash occurs.

Usage:
  PARENT_IMPORT_QT=1 DYLD_PRINT_LIBRARIES=1 python scripts/stress_worker.py > logs/stress-worker-dyld.log 2>&1
"""
import os
import time
import random
from pathlib import Path
from PIL import Image

def main():
    # Optionally import Qt in the parent process to simulate GUI app state.
    if os.environ.get('PARENT_IMPORT_QT') in ('1', 'true', 'True'):
        try:
            import PySide6.QtWidgets  # type: ignore
            print('Parent imported PySide6.QtWidgets')
        except Exception as e:
            print('Failed to import PySide6 in parent:', e)

    # ensure logs dir
    Path('logs').mkdir(exist_ok=True)

    # Import the detection worker lazily
    from src.detection_worker import get_global_worker  # type: ignore

    worker = get_global_worker()
    print('Started global detection worker')

    # Prepare a simple image list: use test_images if present, otherwise create temp images
    img_dir = Path('test_images')
    images = []
    if img_dir.exists() and img_dir.is_dir():
        for p in img_dir.iterdir():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                images.append(str(p))
    # If no images, write a few small blank PNGs
    if not images:
        tmp = Path('logs') / 'stress_tmp_images'
        tmp.mkdir(exist_ok=True)
        for i in range(3):
            path = tmp / f'blank_{i}.png'
            Image.new('RGB', (64, 64), color=(i*30, i*30, i*30)).save(path)
            images.append(str(path))

    print('Images to use for stress:', images)

    # enqueue loop: continuously send detection requests
    reqs = {}
    req_id = 0
    start_time = time.time()
    try:
        while True:
            # small batch: enqueue a few requests
            for _ in range(5):
                img = random.choice(images)
                req_id += 1
                print(f'enqueue {req_id} -> {img}')
                worker.enqueue(img, {})
                reqs[req_id] = img
                time.sleep(0.1)
            # try to collect results for up to 10s
            deadline = time.time() + 10.0
            while reqs and time.time() < deadline:
                # poll worker results
                try:
                    # attempt to get any result by calling get_result for one id
                    some_id = next(iter(reqs))
                    res = worker.get_result(some_id, timeout=2.0)
                    print(f'result {some_id}:', 'error' in res and res.get('error') or f"detections={len(res.get('detections', []))}")
                    reqs.pop(some_id, None)
                except Exception as e:
                    print('get_result polling exception:', e)
                    time.sleep(0.5)
            # short pause
            if time.time() - start_time > 300:
                print('Stress run completed 5 minutes, exiting')
                break
    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        print('Stopping worker')
        try:
            worker.stop()
        except Exception as e:
            print('Failed to stop worker:', e)

    print('Stress runner exiting')


if __name__ == '__main__':
    main()
