#!/usr/bin/env python3
"""Sweep YOLO settings to find parameters that produce >=N person detections for an image.

Usage: python scripts/sweep_find_best_settings.py <image_path> [--out-dir OUT] [--min-count N]

This script uses raw Ultralytics YOLO predictions (no app-enforced floors) and
applies confidence + area filtering to find a configuration that yields at least
`min_count` detections of class `person`.

It saves a JSON report and an overlay PNG with the top detections when found.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_best(image_path: str, out_dir: str, max_sizes, confs, min_area_ratios, min_count: int = 10):
    from ultralytics import YOLO
    model = YOLO('yolov8n')
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    best = None
    report = []

    for imgsz in max_sizes:
        # request a reasonably low conf to get candidate boxes and then filter in-Python
        min_conf_request = min(confs)
        results = model.predict(image_path, imgsz=imgsz, conf=min_conf_request, verbose=False)
        if not results:
            continue
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        if boxes is None:
            continue
        # convert to numpy arrays (boxes xyxy, conf, cls)
        try:
            xyxy = np.asarray(boxes.xyxy.cpu().numpy())
            confs_raw = np.asarray(boxes.conf.cpu().numpy())
            cls_raw = np.asarray(boxes.cls.cpu().numpy()).astype(int)
        except Exception:
            xyxy = np.asarray(boxes.xyxy)
            confs_raw = np.asarray(boxes.conf)
            cls_raw = np.asarray(boxes.cls).astype(int)
        # ultralytics classes are 0-based COCO; person -> class 0 -> app-level 1
        labels = (cls_raw + 1).astype(int)

        for conf_th in confs:
            for min_area in min_area_ratios:
                kept = []
                for b, cf, lab in zip(xyxy, confs_raw, labels):
                    if cf < conf_th:
                        continue
                    # compute area ratio relative to original image area
                    x1, y1, x2, y2 = [float(x) for x in b]
                    # scale coordinates if the detection was produced at a different imgsz
                    # ultralytics returns boxes in original-image coordinates, so area relative to W*H is appropriate
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    area_ratio = area / float(W * H) if (W * H) > 0 else 0.0
                    if area_ratio < min_area:
                        continue
                    if lab == 1:  # person
                        kept.append({'bbox': [x1, y1, x2, y2], 'conf': float(cf)})

                report.append({'imgsz': imgsz, 'conf_th': conf_th, 'min_area': min_area, 'person_count': len(kept)})

                if len(kept) >= min_count and best is None:
                    best = {'imgsz': imgsz, 'conf_th': conf_th, 'min_area': min_area, 'person_count': len(kept), 'people': kept}
                    # we still continue to build report for reproducibility
    # Save report and overlay if found
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    report_path = out_dir_p / (Path(image_path).stem + '.sweep_report.json')
    with report_path.open('w') as fh:
        json.dump({'report': report, 'best': best}, fh, indent=2)

    overlay_path = None
    if best is not None:
        # draw overlay with top-N persons (sorted by conf)
        people = best['people']
        people.sort(key=lambda x: -x['conf'])
        top = people[:min_count]
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
        except Exception:
            font = ImageFont.load_default()
        for i, p in enumerate(top):
            x1, y1, x2, y2 = p['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
            draw.text((x1, y1), f'person {p["conf"]:.2f}', fill='white', font=font)
        overlay_path = out_dir_p / (Path(image_path).stem + '.best.overlay.png')
        draw_img.save(overlay_path)

    return {'report_path': str(report_path), 'best': best, 'overlay_path': str(overlay_path) if overlay_path else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep YOLO settings for person detection')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--out-dir', '-o', default='.cache/visualizations/sweep', help='Directory to save report and overlay')
    parser.add_argument('--min-count', type=int, default=10, help='Minimum person detections to consider success')
    args = parser.parse_args()

    # sensible defaults; adjust if you want broader sweep
    max_sizes = [800, 1280, 1600]
    confs = [0.6, 0.7, 0.8, 0.5, 0.4, 0.3]
    min_area_ratios = [0.01, 0.005, 0.001, 0.0005, 0.0]

    res = find_best(args.image, args.out_dir, max_sizes, confs, min_area_ratios, min_count=args.min_count)
    print('Sweep finished. Report saved to:', res['report_path'])
    if res['best']:
        print('Found best setting:', res['best'])
        if res['overlay_path']:
            print('Saved overlay to', res['overlay_path'])
    else:
        print('No configuration met the min-count requirement')
