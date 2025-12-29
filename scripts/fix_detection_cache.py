#!/usr/bin/env python3
"""
Scan and fix `.cache/object_detections.joblib` by re-running detection
for entries that lack `bbox` or `det_w`/`det_h` fields. Run under poetry:

poetry run python scripts/fix_detection_cache.py --limit 50

"""
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default='.cache/object_detections.joblib')
    parser.add_argument('--root', default='.', help='Root directory to search for image files')
    parser.add_argument('--limit', type=int, default=0, help='Max images to re-detect (0 = no limit)')
    args = parser.parse_args()

    # Defer imports until runtime (avoid heavy imports at module load)
    try:
        from src import object_detection
    except Exception as e:
        print('Failed to import object_detection:', e)
        raise

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print('Cache not found at', cache_path)
        return

    cache = object_detection.load_object_cache()
    # load_object_cache returns dict mapping basename -> detections
    detections = cache
    keys = list(detections.keys())
    fixed = 0
    scanned = 0

    for k in keys:
        scanned += 1
        entry = detections.get(k)
        need = False
        if not entry:
            need = True
        else:
            # entry may be list of tuples or dicts
            if isinstance(entry, list) and len(entry) > 0:
                first = entry[0]
                if not isinstance(first, dict) or 'bbox' not in first or 'det_w' not in first or 'det_h' not in first:
                    need = True
            else:
                need = True

        if not need:
            continue

        print('Re-detecting:', k)
        # Attempt to locate the actual image file by basename under provided root
        img_path = None
        # 1. If k looks like a path and exists, prefer it
        if os.path.isabs(k) and os.path.exists(k):
            img_path = k
        else:
            # 2. Check relative to provided root directly
            candidate = os.path.join(args.root, k)
            if os.path.exists(candidate):
                img_path = candidate
            else:
                # 3. Search recursively under the provided root for matching basename
                for root_dir, dirs, files in os.walk(args.root):
                    if k in files:
                        img_path = os.path.join(root_dir, k)
                        break
        if img_path is None:
            print(' -> image file not found in workspace:', k)
            continue
        try:
            # detect_objects returns list of dicts with bbox/det sizes
            new = object_detection.detect_objects(img_path)
            if new:
                detections[k] = new
                fixed += 1
                print(' -> fixed, found', len(new), 'detections')
            else:
                print(' -> no detections')
        except Exception as e:
            print(' -> detection failed for', k, repr(e))

        if args.limit and fixed >= args.limit:
            break

    # Persist fixed detections back to cache using object_detection API
    try:
        object_detection.save_object_cache(detections)
    except Exception as e:
        print('Failed to save cache via object_detection.save_object_cache:', e)
    print(f'Done. scanned={scanned} fixed={fixed}')

if __name__ == '__main__':
    main()
