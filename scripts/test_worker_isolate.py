#!/usr/bin/env python3
"""Simple script to start a DetectionWorker, send one request, and print results.

This isolates worker startup and a single detection call. It waits for the
`logs/detection_worker_<pid>.ready` marker and prints the worker log tail
so we can see what the child is doing.
"""
import os
import time
import sys

# Adjust this path to a real image on your machine if needed
IMAGE_PATH = os.environ.get('TEST_IMAGE', os.path.join(os.path.expanduser('~/Pictures/photo-dataset'), 'PXL_20250526_190857345.jpg'))

# Ensure logs dir exists so worker can write files
os.makedirs('logs', exist_ok=True)

# Import worker manager
try:
    from src.detection_worker import get_global_worker, stop_global_worker
except Exception as exc:
    print('Failed to import detection worker manager:', exc)
    raise


if __name__ == '__main__':
    print('Starting detection worker (spawn)')
    worker = get_global_worker()
    proc = getattr(worker.proc, 'pid', None)
    print('Worker proc pid:', proc)

    # Wait for ready marker
    ready_path = f'logs/detection_worker_{proc}.ready' if proc else None
    if ready_path:
        print('Waiting for ready marker:', ready_path)
        waited = 0
        while waited < 30:
            if os.path.exists(ready_path):
                print('Ready marker found')
                break
            time.sleep(0.5)
            waited += 0.5
        else:
            print('Timed out waiting for ready marker')
    else:
        print('No worker pid available; continuing without ready check')

    # Enqueue one request and wait
    try:
        print('Enqueuing image:', IMAGE_PATH)
        req_id = worker.enqueue(IMAGE_PATH, {'confidence_threshold': 0.6})
        try:
            qlen = worker.queue_length()
        except Exception:
            qlen = None
        print('Queue length after enqueue:', qlen)

        print('Waiting for result (timeout 180s)')
        try:
            resp = worker.get_result(req_id, timeout=180.0)
            print('Received response for req_id=', req_id)
            print(resp)
        except Exception as exc:
            print('Error or timeout while waiting for worker result:', repr(exc))

    finally:
        try:
            print('Stopping worker')
            stop_global_worker()
        except Exception:
            pass

    # Print last part of the worker log
    if proc:
        log_path = f'logs/detection_worker_{proc}.log'
        print('\n--- Worker log tail (last 200 lines) ---')
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                tail = lines[-200:]
                sys.stdout.writelines(tail)
        except Exception as exc:
            print('Failed to read worker log:', exc)

    print('\nTest script completed')
