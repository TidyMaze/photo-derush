#!/usr/bin/env python3
"""Worker invoked by `check_detection_backends.py` to run a single backend check.
Usage: python check_backend_worker.py <backend> <image_path> <n_iterations>
"""
import json
import sys
import time
import os

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'missing-args'}))
        sys.exit(2)
    backend = sys.argv[1]
    img = sys.argv[2] if sys.argv[2] != 'None' else None
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Optional mode: 'auto' (default), 'cpu', 'mps', 'cuda'
    mode = sys.argv[4] if len(sys.argv) > 4 else 'auto'

    import torch
    print("mps_available:", torch.backends.mps.is_available())
    print("mps_built:", torch.backends.mps.is_built())

    os.environ['DETECTION_BACKEND'] = backend
    rec = {'backend': backend, 'load_ok': False, 'load_error': None, 'load_duration': None, 'device': None, 'inference_ok': None, 'inference_error': None, 'inference_counts': [], 'inference_durations': []}

    try:
        from src import object_detection as od
    except Exception as e:
        rec['load_error'] = f'import_error:{e}'
        print(json.dumps(rec))
        sys.exit(0)

    t0 = time.time()
    try:
        # Use requested mode when loading model; 'auto' lets object_detection pick
        m, w = od._load_model(device=mode if mode in ('cpu', 'mps', 'cuda') else 'auto')
        rec['load_ok'] = True
    except Exception as e:
        rec['load_ok'] = False
        rec['load_error'] = str(e)
        print(json.dumps(rec))
        sys.exit(0)
    t1 = time.time()
    rec['load_duration'] = round(t1-t0,4)
    rec['device'] = od._device

    if img and rec['load_ok']:
        rec['inference_ok'] = True
        for i in range(N):
            t2 = time.time()
            try:
                if backend == 'fasterrcnn':
                    device_arg = 'cpu'
                else:
                    device_arg = mode if mode in ('cpu','mps','cuda') else 'auto'
                dets = od.detect_objects(img, confidence_threshold=0.3, device=device_arg)
                rec['inference_counts'].append(len(dets) if dets is not None else 0)
            except Exception as e:
                rec['inference_ok'] = False
                rec['inference_error'] = str(e)
                break
            t3 = time.time()
            rec['inference_durations'].append(round(t3-t2,4))

    print(json.dumps(rec))
