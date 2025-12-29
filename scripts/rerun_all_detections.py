#!/usr/bin/env python3
"""Purge detection cache and re-run object detection on all images in test_images/.

Saves a JSON report to `.cache/object_detections_report.json` and writes overlays to
`.cache/overlays/` using `src.object_detection` defaults.
"""
import json
import os
from pathlib import Path

from src import object_detection as od


def purge_caches():
    paths = [Path('.cache/object_detections.joblib'), Path('.cache/visualizations'), Path('.cache/overlays')]
    for p in paths:
        if p.exists():
            if p.is_file():
                try:
                    p.unlink()
                    print(f"Removed file {p}")
                except Exception as e:
                    print(f"Failed to remove file {p}: {e}")
            else:
                # remove directory tree
                import shutil
                try:
                    shutil.rmtree(p)
                    print(f"Removed directory {p}")
                except Exception as e:
                    print(f"Failed to remove directory {p}: {e}")


def find_images(root='test_images'):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    rootp = Path(root)
    if not rootp.exists():
        return []
    images = [str(p) for p in rootp.rglob('*') if p.suffix.lower() in exts]
    images.sort()
    return images


def run_all():
    purge_caches()
    images = find_images()
    print(f"Found {len(images)} images")
    if not images:
        return

    # Ensure cache dir exists for saving
    Path('.cache').mkdir(exist_ok=True)

    report = {}
    def progress(i, total, msg):
        print(f"[{i+1}/{total}] {msg}")

    # Use get_objects_for_images which will repopulate cache after purge
    results = od.get_objects_for_images(images, progress_callback=progress)

    # Also create overlays for each image using the current defaults
    overlays = {}
    for img in images:
        basename = os.path.basename(img)
        dets = od.load_object_cache().get(basename) or []
        overlay_path = od.save_overlay(img, detections=dets)
        overlays[basename] = overlay_path

    report['results'] = results
    report['overlays'] = overlays

    out = Path('.cache/object_detections_report.json')
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved report to {out}")


if __name__ == '__main__':
    run_all()
