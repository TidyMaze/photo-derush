"""Check which detection backends load successfully in auto mode on this machine.

This script spawns a fresh Python process for each backend name and measures
how long `_load_model(device='auto')` takes and whether it errors. It uses
DETECTION_BACKEND and DETECTION_DEVICE env variables to force behavior when
needed and prints a concise JSON-like line for each backend.

Run from project root with the project's venv python to get realistic results.
"""
import json
import os
import shlex
import subprocess
import sys
import time
import multiprocessing

BACKENDS = ['fasterrcnn', 'ssdlite', 'yolov8']
TIMEOUT = 300  # seconds per backend (increased to allow multiple inference runs)
PY = sys.executable
# Image to use for a short inference check. Can be overridden by env CHECK_IMAGE.
DEFAULT_CHECK_IMAGE = os.path.join(os.path.expanduser('~/Pictures/photo-dataset'), 'PXL_20250712_183833561.PORTRAIT.jpg')

def run_backend(backend):
    # Use a standalone worker script to avoid pickling issues with multiprocessing.
    worker_py = os.path.join(os.path.dirname(__file__), 'check_backend_worker.py')
    img_path = os.environ.get('CHECK_IMAGE', DEFAULT_CHECK_IMAGE) or 'None'
    n_iter = '3' if backend == 'fasterrcnn' else '10'
    cmd = [PY, worker_py, backend, img_path, n_iter]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TIMEOUT, text=True)
        return p.stdout
    except subprocess.TimeoutExpired:
        return json.dumps({'backend': backend, 'error': 'timeout'})

if __name__ == '__main__':
    results = []
    for b in BACKENDS:
        print(f"Checking backend: {b}")
        out = run_backend(b)
        # Print raw stdout for traceability
        print(out)
        try:
            # Try to find the JSON line in stdout
            last_line = out.strip().splitlines()[-1]
            rec = json.loads(last_line)
        except Exception:
            rec = {'backend': b, 'ok': False, 'error': 'unexpected-output', 'raw': out}
        results.append(rec)

    print('\nSUMMARY:')
    print(json.dumps(results, indent=2))
