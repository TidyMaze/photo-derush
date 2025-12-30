#!/usr/bin/env python3
"""Check why specific groups are separated."""

import sys
import os
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grouping_service import compute_grouping_for_photos, extract_timestamp, extract_camera_id
from src.photo_grouping import PHASH_HAMMING_THRESHOLD


def compute_hash(path: str) -> str:
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


def check_group_separation(group_ids: list[int], image_dir: str):
    """Check why specified groups are separated."""
    
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
        phash_threshold=21,
        progress_reporter=None,
    )
    
    # Organize images by group_id
    groups: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for filename, info in group_info.items():
        gid = info.get('group_id')
        if gid is not None:
            groups[gid].append((filename, info))
    
    # Check each requested group pair
    for i in range(len(group_ids) - 1):
        gid1 = group_ids[i]
        gid2 = group_ids[i + 1]
        
        if gid1 not in groups or gid2 not in groups:
            print(f"\n⚠️  Group {gid1} or {gid2} not found!")
            continue
        
        images1 = groups[gid1]
        images2 = groups[gid2]
        
        print(f"\n{'='*80}")
        print(f"Comparing Group {gid1} ({len(images1)} images) vs Group {gid2} ({len(images2)} images)")
        print(f"{'='*80}")
        
        # Check metadata
        print("\n📊 Metadata Comparison:")
        print(f"\nGroup {gid1}:")
        for fname, info in images1[:3]:  # Show first 3
            path = os.path.join(image_dir, fname)
            ts = extract_timestamp(exif_data.get(fname, {}), path)
            cam = extract_camera_id(exif_data.get(fname, {}))
            print(f"  {fname}:")
            print(f"    session_id={info.get('session_id')}, burst_id={info.get('burst_id')}")
            print(f"    timestamp={ts}, camera={cam}")
        
        print(f"\nGroup {gid2}:")
        for fname, info in images2[:3]:  # Show first 3
            path = os.path.join(image_dir, fname)
            ts = extract_timestamp(exif_data.get(fname, {}), path)
            cam = extract_camera_id(exif_data.get(fname, {}))
            print(f"  {fname}:")
            print(f"    session_id={info.get('session_id')}, burst_id={info.get('burst_id')}")
            print(f"    timestamp={ts}, camera={cam}")
        
        # Check hash distances
        print(f"\n🔍 Hash Distance Analysis (threshold={PHASH_HAMMING_THRESHOLD}):")
        
        # Compute hashes for all images in both groups
        hashes1 = {}
        hashes2 = {}
        
        for fname, _ in images1:
            path = os.path.join(image_dir, fname)
            h = compute_hash(path)
            if h:
                hashes1[fname] = h
        
        for fname, _ in images2:
            path = os.path.join(image_dir, fname)
            h = compute_hash(path)
            if h:
                hashes2[fname] = h
        
        # Find minimum distance between groups
        min_dist = float('inf')
        min_pair = None
        
        for fname1, h1 in hashes1.items():
            for fname2, h2 in hashes2.items():
                dist = hamming_distance(h1, h2)
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (fname1, fname2)
        
        print(f"  Minimum hash distance between groups: {min_dist}")
        if min_pair:
            print(f"  Closest pair: {min_pair[0]} ↔ {min_pair[1]}")
        
        if min_dist <= PHASH_HAMMING_THRESHOLD:
            print(f"  ⚠️  Groups SHOULD be merged (distance {min_dist} <= threshold {PHASH_HAMMING_THRESHOLD})")
        else:
            print(f"  ✅ Groups correctly separated (distance {min_dist} > threshold {PHASH_HAMMING_THRESHOLD})")
        
        # Check if they share burst_id
        burst_ids1 = set(info.get('burst_id') for _, info in images1)
        burst_ids2 = set(info.get('burst_id') for _, info in images2)
        shared_bursts = burst_ids1 & burst_ids2
        
        if shared_bursts:
            print(f"\n⚠️  Groups share burst_id(s): {shared_bursts}")
            print(f"  They SHOULD be merged by burst merging logic!")
        else:
            print(f"\n  Groups have different burst_ids:")
            print(f"    Group {gid1}: {sorted(burst_ids1)}")
            print(f"    Group {gid2}: {sorted(burst_ids2)}")
        
        # Check session_id
        session_ids1 = set(info.get('session_id') for _, info in images1)
        session_ids2 = set(info.get('session_id') for _, info in images2)
        
        if session_ids1 == session_ids2:
            print(f"\n  Same session_id: {session_ids1}")
        else:
            print(f"\n  Different session_ids:")
            print(f"    Group {gid1}: {session_ids1}")
            print(f"    Group {gid2}: {session_ids2}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_group_separation.py <image_dir> [group_id1] [group_id2] ...")
        print("Example: python check_group_separation.py ~/Pictures/photo-dataset 5 6 10 11")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    group_ids = [int(gid) for gid in sys.argv[2:]] if len(sys.argv) > 2 else [5, 6, 10, 11]
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    check_group_separation(group_ids, image_dir)


if __name__ == "__main__":
    main()

