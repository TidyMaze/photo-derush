#!/usr/bin/env python3
"""Threaded repro for YOLOv8 that mimics the GUI streaming workload.

Run with the project's venv python. It will load the YOLO model and run many
predict calls in parallel to try to reproduce native crashes.
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
    parser.add_argument('--count', type=int, default=0, help='0 => all')
    args = parser.parse_args()

    # Force CPU to match our earlier runs
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

    # Gather images
    images_dir = os.path.expanduser(args.images)
    patterns = [os.path.join(images_dir, '*.%s' % ext) for ext in ('jpg','jpeg','png','MP.jpg','JPG')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)
    if args.count and args.count > 0:
        files = files[:args.count]

    print('Found', len(files), 'images; workers=', args.workers)

    # Load model (this may trigger internal fuse/compile paths)
    print('Loading model', model_path)
    model = YOLO(model_path)

    # Warm-up single prediction
    if files:
        print('Warm-up predict on first image')
        try:
            _ = model.predict(files[0], device='cpu', verbose=False)
            print('Warm-up done')
        except Exception as e:
            print('Warm-up failed:', e)

    # Run threaded predictions to increase concurrency stress
    def do_predict(path, idx):
        try:
            print(f'[{idx}] predict {os.path.basename(path)}')
            res = model.predict(path, device='cpu', verbose=False)
            return (idx, 'ok', len(res))
        except Exception as e:
            return (idx, 'error', repr(e))

    if not files:
        print('No images found; exiting')
        return

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_predict, f, i): i for i, f in enumerate(files)}
        try:
            for fut in as_completed(futs):
                results.append(fut.result())
                print('result:', results[-1])
        except Exception as e:
            print('Executor raised:', e)

    elapsed = time.time() - start
    print('Done; elapsed', elapsed)

if __name__ == '__main__':
    main()
