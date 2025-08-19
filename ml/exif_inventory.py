import json
from pathlib import Path
from collections import Counter
import sys
import argparse
import logging

try:
    from PIL import Image  # noqa: F401  (kept to fail early if PIL missing)
except Exception:  # pragma: no cover
    Image = None

from photo_derush.utils import extract_exif  # centralized EXIF loader

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Interesting EXIF tag names to explicitly report (aligned with features)
KEY_TAGS = [
    'ExposureBiasValue','ExposureMode','WhiteBalance','SceneCaptureType','Saturation',
    'Contrast','Sharpness','DigitalZoomRatio','FocalLengthIn35mmFilm','CompositeImage',
    'LensMake','LensModel','SubjectDistance','FocalLength','ExposureTime','FNumber',
    'ApertureValue','ISO','ISOSpeedRatings','PhotographicSensitivity'
]

IMAGE_EXTS = {'.jpg','.jpeg','.png','.tif','.tiff','.webp','.bmp','.gif'}


def iter_images(root: Path, recursive: bool):
    if not root.exists():
        return
    if recursive:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def inventory(dir_path: Path, recursive: bool, limit: int | None = None):
    name_counter = Counter()
    sample_values = {}
    per_image_tag_counts = []
    images_with_exif = 0
    total_images = 0
    key_tag_presence = {k: 0 for k in KEY_TAGS}
    images_scanned = 0
    for img_path in iter_images(dir_path, recursive):
        total_images += 1
        if limit and total_images > limit:
            break
        tagmap = extract_exif(str(img_path))  # central call
        if tagmap:
            images_with_exif += 1
            per_image_tag_counts.append(len(tagmap))
            for name, value in tagmap.items():
                name_counter[name] += 1
                if name not in sample_values:
                    sv = value
                    if isinstance(sv, (bytes, bytearray)):
                        sv = sv[:16]
                    sample_values[name] = sv
            for kt in key_tag_presence:
                if kt in tagmap:
                    key_tag_presence[kt] += 1
        images_scanned += 1
    return {
        'directory': str(dir_path),
        'recursive': recursive,
        'limit': limit,
        'images_scanned': images_scanned,
        'images_with_exif': images_with_exif,
        'unique_tag_names': len(name_counter),
        'tag_frequency': name_counter.most_common(),
        'sample_values': sample_values,
        'key_tag_presence': key_tag_presence,
        'per_image_tag_count_min': min(per_image_tag_counts) if per_image_tag_counts else 0,
        'per_image_tag_count_max': max(per_image_tag_counts) if per_image_tag_counts else 0,
    }


def format_human(inv: dict, top: int):
    lines = []
    lines.append(f"Directory: {inv['directory']}")
    lines.append(f"Images scanned: {inv['images_scanned']} (with EXIF: {inv['images_with_exif']})")
    lines.append(f"Unique EXIF tag names: {inv['unique_tag_names']}")
    lines.append("Key tag presence (images containing tag / with EXIF):")
    iw = inv['images_with_exif'] or 1
    for k, c in inv['key_tag_presence'].items():
        pct = (c / iw) * 100.0 if iw else 0.0
        lines.append(f"  - {k}: {c} ({pct:.1f}%)")
    lines.append(f"Tag count range per EXIF-bearing image: {inv['per_image_tag_count_min']}-{inv['per_image_tag_count_max']}")
    lines.append(f"\nTop {top} tags:")
    for name, cnt in inv['tag_frequency'][:top]:
        sv = inv['sample_values'].get(name)
        lines.append(f"  {name}: {cnt} sample={sv}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='EXIF inventory (centralized extract_exif).')
    parser.add_argument('--dir', type=Path, required=False, default=Path('/Users/yannrolland/Pictures/photo-dataset'), help='Directory of images')
    parser.add_argument('-r','--recursive', action='store_true', help='Recurse into subdirectories')
    parser.add_argument('--limit', type=int, help='Max images to scan')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--top', type=int, default=40, help='Show top N tags (human mode)')
    parser.add_argument('--out', type=Path, help='Write full JSON to this path')
    args = parser.parse_args()

    if Image is None:
        print('PIL not available')
        sys.exit(1)
    if not args.dir.exists():
        print(f"Directory not found: {args.dir}")
        sys.exit(2)

    inv = inventory(args.dir, args.recursive, args.limit)
    if args.json:
        out_json = json.dumps(inv, indent=2, default=str)
        print(out_json)
    else:
        print(format_human(inv, args.top))
    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(inv, f, indent=2, default=str)
        except Exception as e:  # pragma: no cover
            logger.warning('Failed writing output JSON: %s', e)

if __name__ == '__main__':
    main()
