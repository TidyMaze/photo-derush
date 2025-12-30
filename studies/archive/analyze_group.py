#!/usr/bin/env python3
"""Analyze a specific group to understand why images are grouped together."""

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos, extract_camera_id, extract_timestamp
from src.photo_grouping import PHASH_HAMMING_THRESHOLD


def compute_hash(path: str) -> str | None:
    """Compute perceptual hash for an image."""
    try:
        with Image.open(path) as img:
            hash_obj = imagehash.phash(img)
            return str(hash_obj)
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    """Calculate Hamming distance between two hash strings."""
    if h1 is None or h2 is None:
        return float('inf')
    try:
        hash1 = imagehash.hex_to_hash(h1)
        hash2 = imagehash.hex_to_hash(h2)
        return hash1 - hash2
    except Exception:
        return float('inf')


def analyze_group(group_id: int, image_dir: str):
    """Analyze why images in a specific group are grouped together."""

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]

    if not filenames:
        print("No images found!")
        return

    # Load EXIF data
    exif_data = {}
    keep_probabilities = {}
    for fname in filenames:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
            exif = img.getexif()
            exif_data[fname] = {}
            if exif:
                for tag_id, value in exif.items():
                    tag = img.getexif().get(tag_id)
                    if tag:
                        exif_data[fname][str(tag)] = str(value)
            keep_probabilities[fname] = 0.5
        except Exception:
            exif_data[fname] = {}
            keep_probabilities[fname] = 0.5

    print("Computing grouping...")
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=keep_probabilities,
        quality_metrics=None,
        session_gap_min=10,
        burst_gap_sec=15.0,
        phash_threshold=PHASH_HAMMING_THRESHOLD,
        progress_reporter=None,
    )

    # Find images in the specified group
    group_images = []
    for filename, info in group_info.items():
        if info.get('group_id') == group_id:
            group_images.append((filename, info))

    if not group_images:
        print(f"Group #{group_id} not found!")
        return

    print(f"\n{'='*80}")
    print(f"Group #{group_id} Analysis ({len(group_images)} images)")
    print(f"{'='*80}\n")

    # Sort by filename for consistent output
    group_images.sort(key=lambda x: x[0])

    # Extract metadata
    print("📊 Metadata Summary:")
    print("-" * 80)

    session_ids = set()
    burst_ids = set()
    cameras = set()
    timestamps = []

    for filename, info in group_images:
        session_id = info.get('session_id')
        burst_id = info.get('burst_id')
        if session_id is not None:
            session_ids.add(session_id)
        if burst_id is not None:
            burst_ids.add(burst_id)

        path = os.path.join(image_dir, filename)
        ts = extract_timestamp(exif_data.get(filename, {}), path)
        cam = extract_camera_id(exif_data.get(filename, {}))
        timestamps.append((filename, ts))
        if cam:
            cameras.add(cam)

    print(f"  Session IDs: {sorted(session_ids)}")
    print(f"  Burst IDs: {sorted(burst_ids)}")
    print(f"  Cameras: {sorted(cameras) if cameras else ['unknown']}")

    if timestamps:
        timestamps.sort(key=lambda x: x[1])
        print("\n  Time range:")
        print(f"    Earliest: {timestamps[0][1]} ({timestamps[0][0]})")
        print(f"    Latest: {timestamps[-1][1]} ({timestamps[-1][0]})")
        if len(timestamps) > 1:
            time_span = (timestamps[-1][1] - timestamps[0][1]).total_seconds()
            print(f"    Span: {time_span:.1f} seconds ({time_span/60:.1f} minutes)")

    # Hash distance analysis
    print(f"\n🔍 Hash Distance Analysis (threshold={PHASH_HAMMING_THRESHOLD}):")
    print("-" * 80)

    # Compute hashes
    hashes = {}
    for filename, _ in group_images:
        path = os.path.join(image_dir, filename)
        h = compute_hash(path)
        if h:
            hashes[filename] = h

    if len(hashes) < 2:
        print("  Not enough images with valid hashes for distance analysis")
    else:
        # Calculate all pairwise distances
        distances = []
        hash_list = list(hashes.items())
        for i, (f1, h1) in enumerate(hash_list):
            for j, (f2, h2) in enumerate(hash_list[i+1:], i+1):
                dist = hamming_distance(h1, h2)
                distances.append((dist, f1, f2))

        distances.sort()

        print("  Pairwise distances (showing first 10):")
        for dist, f1, f2 in distances[:10]:
            print(f"    {os.path.basename(f1)} ↔ {os.path.basename(f2)}: {dist}")

        if distances:
            min_dist = distances[0][0]
            max_dist = distances[-1][0]
            avg_dist = sum(d[0] for d in distances) / len(distances)
            print("\n  Statistics:")
            print(f"    Min distance: {min_dist}")
            print(f"    Max distance: {max_dist}")
            print(f"    Avg distance: {avg_dist:.1f}")

            if max_dist <= PHASH_HAMMING_THRESHOLD:
                print(f"  ✅ All pairs within threshold ({max_dist} <= {PHASH_HAMMING_THRESHOLD})")
            else:
                print(f"  ⚠️  Some pairs exceed threshold (max {max_dist} > {PHASH_HAMMING_THRESHOLD})")
                print("     This suggests grouping may be due to burst merging, not hash similarity")

    # Check burst merging
    print("\n📸 Burst Analysis:")
    print("-" * 80)

    if len(burst_ids) == 1:
        print(f"  All images share burst_id={list(burst_ids)[0]}")
        print("  ✅ Grouped by burst merging (same burst)")
    elif len(burst_ids) > 1:
        print(f"  Images span {len(burst_ids)} different bursts: {sorted(burst_ids)}")
        print("  ⚠️  Grouped despite different bursts - likely hash similarity")

        # Show which images are in which bursts
        burst_to_images = defaultdict(list)
        for filename, info in group_images:
            burst_id = info.get('burst_id')
            if burst_id is not None:
                burst_to_images[burst_id].append(filename)

        for burst_id, images in sorted(burst_to_images.items()):
            print(f"\n    Burst {burst_id} ({len(images)} images):")
            for img in sorted(images)[:5]:
                print(f"      {os.path.basename(img)}")
            if len(images) > 5:
                print(f"      ... and {len(images) - 5} more")

    # Detailed image list with EXIF/metadata source
    print(f"\n📋 Images in Group #{group_id} (with metadata source):")
    print("-" * 80)
    for idx, (filename, info) in enumerate(group_images, 1):
        path = os.path.join(image_dir, filename)
        exif = exif_data.get(filename, {})

        # Determine timestamp source
        ts_source = "unknown"
        dt_original = None

        # Check provided EXIF dict
        if isinstance(exif, dict):
            dt_original = exif.get("DateTimeOriginal") or exif.get(36867) or exif.get("36867")
            if dt_original:
                ts_source = "EXIF DateTimeOriginal (from dict)"
            else:
                dt_original = exif.get("DateTime") or exif.get(306) or exif.get("306")
                if dt_original:
                    ts_source = "EXIF DateTime (from dict)"

        # Check file directly
        if not dt_original:
            try:
                from PIL import Image
                from PIL.ExifTags import EXIFIFD
                with Image.open(path) as img:
                    exif_obj = img.getexif()
                    if exif_obj:
                        try:
                            exif_ifd = exif_obj.get_ifd(EXIFIFD)
                            dt_original = exif_ifd.get(36867)  # DateTimeOriginal
                            if dt_original:
                                ts_source = "EXIF DateTimeOriginal (from file)"
                        except Exception:
                            pass
                        if not dt_original:
                            dt_original = exif_obj.get(306)  # DateTime
                            if dt_original:
                                ts_source = "EXIF DateTime (from file)"
            except Exception:
                pass

        # Check if fallback to mtime
        if not dt_original:
            try:
                mtime = os.path.getmtime(path)
                ts_source = f"File mtime (fallback): {datetime.fromtimestamp(mtime)}"
            except Exception:
                ts_source = "datetime.now() (final fallback)"

        ts = extract_timestamp(exif, path)
        cam = extract_camera_id(exif)
        session_id = info.get('session_id')
        burst_id = info.get('burst_id')
        is_best = info.get('is_group_best', False)

        print(f"\n  {idx}. {os.path.basename(filename)}")
        print(f"     Session: {session_id}, Burst: {burst_id}")
        print(f"     Timestamp used: {ts} (source: {ts_source})")

        # Show camera info
        make = exif.get("Make") or exif.get("271") or ""
        model = exif.get("Model") or exif.get("272") or ""
        if make or model:
            print(f"     Camera: {make} {model}".strip())
        else:
            print("     Camera: unknown (no EXIF Make/Model)")

        # Show EXIF availability
        if exif:
            exif_keys = list(exif.keys())
            print(f"     EXIF fields found: {len(exif_keys)} ({', '.join(sorted(exif_keys)[:5])}{'...' if len(exif_keys) > 5 else ''})")
        else:
            print("     EXIF: None (empty dict)")

        # Check if file has EXIF when read directly
        try:
            from PIL import Image
            with Image.open(path) as img:
                exif_obj = img.getexif()
                if exif_obj:
                    exif_count = len(exif_obj)
                    print(f"     EXIF in file: {exif_count} tags")
                else:
                    print("     EXIF in file: None")
        except Exception as e:
            print(f"     EXIF in file: Error reading ({type(e).__name__})")

        if is_best:
            print("     ⭐ BEST PICK")

    print(f"\n{'='*80}")
    print(f"Summary: Group #{group_id} contains {len(group_images)} images")
    if len(burst_ids) == 1:
        print(f"Reason: All images share the same burst_id ({list(burst_ids)[0]})")
    elif len(hashes) >= 2 and distances:
        max_dist = max(d[0] for d in distances)
        if max_dist <= PHASH_HAMMING_THRESHOLD:
            print(f"Reason: Hash similarity (max distance {max_dist} <= threshold {PHASH_HAMMING_THRESHOLD})")
        else:
            print("Reason: Mixed (hash similarity + burst merging)")
    else:
        print("Reason: Unable to determine (insufficient hash data)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_group.py <image_dir> <group_id>")
        print("Example: python analyze_group.py ~/Pictures/photo-dataset 493")
        sys.exit(1)

    image_dir = sys.argv[1]
    try:
        group_id = int(sys.argv[2])
    except ValueError:
        print(f"Error: group_id must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)

    analyze_group(group_id, image_dir)


if __name__ == "__main__":
    main()

