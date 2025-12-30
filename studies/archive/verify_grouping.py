#!/usr/bin/env python3
"""Verify grouping correctness by checking hash distances within groups."""

import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import imagehash
from PIL import Image
from src.grouping_service import compute_grouping_for_photos
from src.repository import RatingsTagsRepository
from src.model import ImageModel


def load_last_dir():
    """Load last directory from config."""
    config_file = Path.home() / ".photo-derush-cache" / "last_dir.txt"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                directory = f.read().strip()
                if os.path.isdir(directory):
                    return directory
        except Exception:
            pass
    return os.path.expanduser('~/Pictures/photo-dataset')


def verify_grouping(image_dir: str):
    """Verify grouping correctness by checking hash distances."""
    print(f"Loading images from: {image_dir}")
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        f for f in os.listdir(image_dir)
        if any(f.endswith(ext) for ext in image_extensions)
    ]
    
    if not filenames:
        print("No images found!")
        return
    
    print(f"Found {len(filenames)} images")
    
    # Dummy data for grouping
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
        except Exception as e:
            exif_data[fname] = {}
            keep_probabilities[fname] = 0.5
    
    print("Computing grouping...")
    group_info = compute_grouping_for_photos(
        filenames=filenames,
        image_dir=image_dir,
        exif_data=exif_data,
        keep_probabilities=keep_probabilities,
        quality_metrics=None,
        session_gap_min=30,
        burst_gap_sec=1.0,
        phash_threshold=8,
        progress_reporter=None,
    )
    
    # Organize by group
    groups_by_id = defaultdict(list)
    for filename, info in group_info.items():
        group_id = info.get("group_id")
        if group_id is not None:
            groups_by_id[group_id].append((filename, info))
    
    # Sort by size
    sorted_groups = sorted(groups_by_id.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Top 10 Groups by Size")
    print(f"{'='*80}")
    
    # Compute hash distances within groups
    hash_cache = {}
    
    def get_hash(fname: str) -> imagehash.ImageHash | None:
        if fname in hash_cache:
            return hash_cache[fname]
        path = os.path.join(image_dir, fname)
        try:
            with Image.open(path) as img:
                phash = imagehash.phash(img)
                hash_cache[fname] = phash
                return phash
        except Exception:
            return None
    
    issues = []
    
    for i, (group_id, images) in enumerate(sorted_groups[:10], 1):
        print(f"\nGroup {group_id}: {len(images)} images")
        
        # Check hash distances within group
        hashes = {}
        for filename, info in images:
            h = get_hash(filename)
            if h:
                hashes[filename] = h
        
        if len(hashes) < 2:
            print(f"  ⚠️  Only {len(hashes)} valid hash(es) - cannot verify distances")
            continue
        
        # Check all pairs
        max_distance = 0
        min_distance = float('inf')
        distances = []
        
        filenames_list = list(hashes.keys())
        for j, fname1 in enumerate(filenames_list):
            h1 = hashes[fname1]
            for fname2 in filenames_list[j+1:]:
                h2 = hashes[fname2]
                dist = h1 - h2
                distances.append((fname1, fname2, dist))
                max_distance = max(max_distance, dist)
                min_distance = min(min_distance, dist)
        
        print(f"  Hash distances: min={min_distance}, max={max_distance}, threshold=8")
        
        if max_distance > 8:
            issues.append((group_id, max_distance, len(images)))
            print(f"  ❌ MAX DISTANCE {max_distance} EXCEEDS THRESHOLD 8!")
            print(f"     This group should be split (transitive connection issue)")
        
        # Show best pick
        best_image = next((f for f, info in images if info.get("is_group_best")), None)
        if best_image:
            print(f"  Best pick: {os.path.basename(best_image)}")
        
        # Show sample filenames
        print(f"  Sample images:")
        for filename, info in images[:3]:
            print(f"    - {os.path.basename(filename)}")
        if len(images) > 3:
            print(f"    ... and {len(images) - 3} more")
    
    if issues:
        print(f"\n{'='*80}")
        print(f"⚠️  GROUPING ISSUES FOUND:")
        print(f"{'='*80}")
        for group_id, max_dist, size in issues:
            print(f"  Group {group_id} (size={size}): max distance {max_dist} > threshold 12")
        print(f"\nThis suggests transitive connections are creating oversized groups.")
        print(f"Consider reducing threshold or using a different grouping strategy.")
    else:
        print(f"\n{'='*80}")
        print(f"✅ All top 10 groups have hash distances <= threshold 8")
        print(f"{'='*80}")
    
    return group_info


if __name__ == "__main__":
    image_dir = load_last_dir()
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        sys.exit(1)
    
    verify_grouping(image_dir)

