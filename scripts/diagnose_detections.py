#!/usr/bin/env python3
"""Spawn-safe diagnostic script to run detection worker and collect per-image results.

Run from the repo root: `poetry run python scripts/diagnose_detections.py`
"""
import json
import os
from pathlib import Path
import time

from src import detection_worker, object_detection


def main():
    out_dir = Path('.cache/detection_diagnostics')
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(os.environ.get('PHOTO_DATASET_DIR', os.path.expanduser('~/Pictures/photo-dataset')))

    cache = object_detection.load_object_cache()
    missing = [k for k, v in cache.items() if not v]
    if not missing:
        all_files = [p.name for p in base_dir.glob('*') if p.is_file()]
        missing = [f for f in all_files if f not in cache]

    sel = missing[:10]
    print('Will diagnose', len(sel), 'files')
    if not sel:
        print('No candidates to diagnose; exiting')
        return 0

    # Ensure worker mode is enabled for this process
    os.environ['DETECTION_WORKER'] = '1'

    # Start worker
    try:
        w = detection_worker.get_global_worker()
    except Exception as e:
        print('Failed to start detection worker:', e)
        return 2

    print('Worker started, pid=', getattr(w.proc, 'pid', None))

    for name in sel:
        p = base_dir / name
        if not p.exists():
            print('File not found', p)
            continue
        print('Enqueue', name)
        try:
            req_id = w.enqueue(str(p), {'confidence_threshold': 0.0, 'min_area_ratio': 0.0})
            resp = w.get_result(req_id, timeout=60.0)
        except Exception as e:
            print('Worker request failed for', name, '->', e)
            resp = {'error': str(e)}

        # Also call detect_objects default (best-effort)
        try:
            det_default = object_detection.detect_objects(str(p))
        except Exception as e:
            det_default = []
            print('detect_objects default crashed for', name, '->', e)

        out = {
            'filename': name,
            'worker_resp': resp,
            'detect_default': det_default,
        }
        fp = out_dir / (name + '.json')
        fp.write_text(json.dumps(out, indent=2))
        print('Wrote', fp)

    # Stop worker cleanly
    try:
        detection_worker.stop_global_worker()
    except Exception:
        pass

    print('Diagnostics complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
