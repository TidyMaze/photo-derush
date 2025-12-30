#!/usr/bin/env python3
"""Compare two groups to see why they're separate and if they should merge."""

import os
import sys
from pathlib import Path

import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos
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


def compare_groups(group_id1: int, group_id2: int, image_dir: str):
    """Compare two groups to see why they're separate."""

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

    # Find images in both groups
    group1_images = [(f, info) for f, info in group_info.items() if info.get('group_id') == group_id1]
    group2_images = [(f, info) for f, info in group_info.items() if info.get('group_id') == group_id2]

    if not group1_images:
        print(f"Group #{group_id1} not found!")
        return
    if not group2_images:
        print(f"Group #{group_id2} not found!")
        return

    print(f"\n{'='*80}")
    print(f"Comparing Group #{group_id1} ({len(group1_images)} images) vs Group #{group_id2} ({len(group2_images)} images)")
    print(f"{'='*80}\n")

    # Compute hashes for all images
    hashes1 = {}
    hashes2 = {}

    for filename, _ in group1_images:
        path = os.path.join(image_dir, filename)
        h = compute_hash(path)
        if h:
            hashes1[filename] = h

    for filename, _ in group2_images:
        path = os.path.join(image_dir, filename)
        h = compute_hash(path)
        if h:
            hashes2[filename] = h

    # Find minimum distance between groups
    min_dist = float('inf')
    min_pair = None

    for f1, h1 in hashes1.items():
        for f2, h2 in hashes2.items():
            dist = hamming_distance(h1, h2)
            if dist < min_dist:
                min_dist = dist
                min_pair = (f1, f2)

    print("🔍 Hash Distance Between Groups:")
    print(f"  Minimum distance: {min_dist}")
    if min_pair:
        print(f"  Closest pair: {os.path.basename(min_pair[0])} ↔ {os.path.basename(min_pair[1])}")

    print(f"\n  Threshold: {PHASH_HAMMING_THRESHOLD}")
    if min_dist <= PHASH_HAMMING_THRESHOLD:
        print(f"  ✅ Groups SHOULD be merged (distance {min_dist} <= threshold {PHASH_HAMMING_THRESHOLD})")
    else:
        print(f"  ⚠️  Groups correctly separated (distance {min_dist} > threshold {PHASH_HAMMING_THRESHOLD})")
        print(f"     To merge them, threshold would need to be at least {min_dist}")

    # Check burst/session overlap
    burst_ids1 = set(info.get('burst_id') for _, info in group1_images)
    burst_ids2 = set(info.get('burst_id') for _, info in group2_images)
    shared_bursts = burst_ids1 & burst_ids2

    session_ids1 = set(info.get('session_id') for _, info in group1_images)
    session_ids2 = set(info.get('session_id') for _, info in group2_images)
    shared_sessions = session_ids1 & session_ids2

    print("\n📸 Burst/Session Overlap:")
    print(f"  Group #{group_id1} bursts: {sorted(burst_ids1)}")
    print(f"  Group #{group_id2} bursts: {sorted(burst_ids2)}")
    if shared_bursts:
        print(f"  ✅ Share burst(s): {shared_bursts}")
    else:
        print("  ❌ No shared bursts")

    print(f"  Group #{group_id1} sessions: {sorted(session_ids1)}")
    print(f"  Group #{group_id2} sessions: {sorted(session_ids2)}")
    if shared_sessions:
        print(f"  ✅ Share session(s): {shared_sessions}")
    else:
        print("  ❌ No shared sessions")

    # Show sample images
    print("\n📋 Sample Images:")
    print(f"  Group #{group_id1} (first 3):")
    for f, _ in group1_images[:3]:
        print(f"    {os.path.basename(f)}")
    print(f"  Group #{group_id2} (first 3):")
    for f, _ in group2_images[:3]:
        print(f"    {os.path.basename(f)}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_two_groups.py <image_dir> <group_id1> <group_id2>")
        print("Example: python compare_two_groups.py ~/Pictures/photo-dataset 0 1")
        sys.exit(1)

    image_dir = sys.argv[1]
    try:
        group_id1 = int(sys.argv[2])
        group_id2 = int(sys.argv[3])
    except ValueError:
        print("Error: group_ids must be integers")
        sys.exit(1)

    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)

    compare_groups(group_id1, group_id2, image_dir)


if __name__ == "__main__":
    main()

