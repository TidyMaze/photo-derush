#!/usr/bin/env python3
"""Streaming-style repro for YOLOv8 that mimics GUI workload.

This script loads a single YOLO model instance and starts N worker threads.
Each worker loops over the image list for `loops` iterations and calls
`model.predict(image, device='cpu')` — sharing the same model instance across
threads to mimic how the GUI schedules detection tasks.
"""
import os
import sys
import time
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--images', default='test_images')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--loops', type=int, default=10)
    parser.add_argument('--count', type=int, default=0, help='0 => all')
    args = parser.parse_args()

    os.environ.setdefault('DETECTION_DEVICE', 'cpu')

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print('Failed to import ultralytics.YOLO:', exc)
        sys.exit(2)

    model_path = args.model
    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        sys.exit(2)

    images_dir = os.path.expanduser(args.images)
    patterns = [os.path.join(images_dir, '*.%s' % ext) for ext in ('jpg','jpeg','png','MP.jpg','JPG')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)
    if args.count and args.count > 0:
        files = files[:args.count]

    if not files:
        print('No images found; exiting')
        return

    print('Found', len(files), 'images; workers=', args.workers, 'loops=', args.loops)

    print('Loading model', model_path)
    model = YOLO(model_path)

    # Warmup
    try:
        print('Warm-up predict on first image')
        _ = model.predict(files[0], device='cpu', verbose=False)
        print('Warm-up done')
    except Exception as e:
        print('Warm-up failed:', e)

    def worker_loop(worker_id):
        for loop in range(args.loops):
            for i, f in enumerate(files):
                try:
                    # perform predict; share model instance
                    _ = model.predict(f, device='cpu', verbose=False)
                except Exception as e:
                    print(f'worker {worker_id} loop {loop} file {i} error: {e}')
                    # re-raise to make failure obvious
                    raise
            print(f'worker {worker_id} completed loop {loop}')
        return worker_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker_loop, wid) for wid in range(args.workers)]
        try:
            for fut in as_completed(futures):
                print('worker done:', fut.result())
        except Exception as e:
            print('executor exception:', e)

    print('Done; elapsed', time.time() - start)

if __name__ == '__main__':
    main()
